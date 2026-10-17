"""Offline tests: không gọi API, không cần GPU hoặc API key thật."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main_v4
from configs import config_v4
from src.judge_v4 import GemmaJudgeV4
from src.v4_validation import SCORE_NAMES, normalize_evidence


def sample():
    return {"content": "Sự kiện ở Hà Nội.", "question": "Sự kiện ở đâu?",
            "answer": "Sự kiện ở Huế, do X tài trợ.", "label": "mixed_hallucination",
            "chunk_id": "chunk1"}


def evidence():
    return {
        "is_evaluable": True, "question_status": "answerable",
        "contradicted_claims": [{"answer_span": "Sự kiện ở Huế", "context_evidence": "Sự kiện ở Hà Nội."}],
        "unsupported_claims": [{"answer_span": "do X tài trợ"}],
        "proposed_label": "mixed_hallucination", "alternative_label": None,
        "self_contradiction": False,
        "scores": {**{name: 5 for name in SCORE_NAMES}, "total_score": 7},
        "uncertainty_factors": [], "confidence": 0.98,
        "short_reason": "Địa điểm sai và nhà tài trợ không có căn cứ.",
    }


def normalize(raw):
    item = sample()
    return normalize_evidence(raw, item["label"], item["content"], item["answer"])


class EvidenceTests(unittest.TestCase):
    def test_mixed_and_total(self):
        result = normalize(evidence())
        self.assertEqual(result["actual_label"], "mixed_hallucination")
        self.assertEqual(result["scores"]["total_score"], 30)
        self.assertEqual(result["confidence"], 0.98)
        self.assertEqual(result["confidence_source"], "model_verbalized_uncalibrated")
        self.assertNotIn("needs_review", result)

    def test_duplicate_cross_type_rejected(self):
        raw = evidence()
        raw["unsupported_claims"] = [{"answer_span": "Sự kiện ở Huế"}]
        with self.assertRaisesRegex(ValueError, "Claim trùng"):
            normalize(raw)

    def test_duplicate_same_type_rejected(self):
        raw = evidence()
        raw["unsupported_claims"] *= 2
        with self.assertRaisesRegex(ValueError, "Claim trùng"):
            normalize(raw)

    def test_missing_array_not_silently_no_hallu(self):
        raw = evidence()
        del raw["contradicted_claims"]
        with self.assertRaises(ValueError):
            normalize(raw)

    def test_quote_must_exist(self):
        raw = evidence()
        raw["contradicted_claims"][0]["context_evidence"] = "Sự kiện ở Đà Nẵng."
        with self.assertRaisesRegex(ValueError, "context_evidence"):
            normalize(raw)

    def test_quality_error_preserves_factual(self):
        for bad in (None, {"plausibility": True}, {"fluency": 6}):
            raw = evidence()
            raw["scores"] = bad
            result = normalize(raw)
            self.assertEqual(result["actual_label"], "mixed_hallucination")
            self.assertEqual(result["quality_status"], "error")
            self.assertIsNone(result["scores"])
            self.assertEqual(result["raw_scores"], bad)

    def test_no_hallu(self):
        raw = evidence()
        raw.update(contradicted_claims=[], unsupported_claims=[], proposed_label="no_hallucination")
        result = normalize(raw)
        self.assertFalse(result["has_hallucination"])
        self.assertFalse(result["label_match"])

    def test_mismatched_proposal_rejected(self):
        raw = evidence()
        raw["proposed_label"] = "no_hallucination"
        with self.assertRaisesRegex(ValueError, "không khớp evidence"):
            normalize(raw)

    def test_cross_type_overlap_rejected(self):
        raw = evidence()
        raw["unsupported_claims"] = [{"answer_span": "Sự kiện ở Huế, do X tài trợ"}]
        with self.assertRaisesRegex(ValueError, "chồng lấn"):
            normalize(raw)

    def test_invalid_requires_empty_claims(self):
        raw = evidence()
        raw.update(is_evaluable=False, proposed_label="invalid", confidence=0.5)
        with self.assertRaisesRegex(ValueError, "claims rỗng"):
            normalize(raw)

    def test_confidence_allows_values_above_point_95(self):
        raw = evidence()
        raw["confidence"] = 1.0
        self.assertEqual(normalize(raw)["confidence"], 1.0)

    def test_alternative_label_requires_uncertainty(self):
        raw = evidence()
        raw["alternative_label"] = "intrinsic_hallucination"
        with self.assertRaisesRegex(ValueError, "uncertainty_factors"):
            normalize(raw)

        raw["uncertainty_factors"] = ["multiple_reasonable_labels"]
        with self.assertRaisesRegex(ValueError, "confidence"):
            normalize(raw)
        raw["confidence"] = 0.84
        self.assertEqual(normalize(raw)["alternative_label"], "intrinsic_hallucination")

    def test_unknown_uncertainty_factor_rejected(self):
        raw = evidence()
        raw["uncertainty_factors"] = ["not_a_valid_factor"]
        with self.assertRaisesRegex(ValueError, "không hợp lệ"):
            normalize(raw)

    def test_uncertainty_caps_confidence_without_alternative(self):
        raw = evidence()
        raw["uncertainty_factors"] = ["evidence_requires_nontrivial_inference"]
        raw["confidence"] = 0.90
        with self.assertRaisesRegex(ValueError, "uncertainty_factors"):
            normalize(raw)
        raw["confidence"] = 0.84
        self.assertEqual(normalize(raw)["confidence"], 0.84)


class JudgeTests(unittest.TestCase):
    def setUp(self):
        cfg = SimpleNamespace(**{k: v for k, v in vars(config_v4).items() if k.isupper()})
        cfg.API_KEY = "offline-test"
        cfg.MAX_RETRIES = 1
        self.judge = GemmaJudgeV4(cfg, cfg.SYSTEM_PROMPT_PATH, cfg.USER_PROMPT_PATH)
        self.addCleanup(self.judge.session.close)

    def run_evidence(self, raw):
        response = {"choices": [{"message": {"content": json.dumps(raw)}}], "usage": {"total_tokens": 10}}
        with patch.object(self.judge, "_call_api", return_value=response):
            return self.judge.judge(main_v4.normalize(sample()))

    def test_provider_price_ceiling_sent_in_request(self):
        self.judge.config.PROVIDER_MAX_PRICE = {"prompt": 0.20, "completion": 0.50}
        response = SimpleNamespace(status_code=200, ok=True)
        response.json = lambda: {"choices": [{"message": {"content": "{}"}}]}
        with patch.object(self.judge.session, "post", return_value=response) as post:
            self.judge._call_api([{"role": "user", "content": "test"}])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["provider"], {
            "sort": "price",
            "max_price": {"prompt": 0.20, "completion": 0.50},
        })

    def test_model_confidence_preserved_without_routing(self):
        result = self.run_evidence(evidence())
        self.assertTrue(result["success"])
        value = result["evaluation"]
        self.assertEqual(value["confidence"], 0.98)
        self.assertEqual(value["confidence_source"], "model_verbalized_uncalibrated")
        self.assertNotIn("routing_decision", value)
        self.assertNotIn("needs_review", value)
        self.assertNotIn("classifier_label", value)

    def test_quality_error_not_sample_failure(self):
        raw = evidence()
        raw["scores"] = None
        result = self.run_evidence(raw)
        self.assertTrue(result["success"])
        self.assertEqual(result["evaluation"]["quality_status"], "error")

    def test_invalid_schema_preserves_raw_output_on_failure(self):
        raw = evidence()
        raw["unsupported_claims"] = [{"answer_span": "Sự kiện ở Huế"}]
        result = self.run_evidence(raw)
        self.assertFalse(result["success"])
        self.assertEqual(json.loads(result["raw_output"]), raw)

    def test_no_original_label_in_prompt(self):
        text = self.judge._build_user_prompt("literal {answer}", "Q", "my answer", "secret-label")
        self.assertIn("literal {answer}", text)
        self.assertNotIn("secret-label", text)


class ResumeTests(unittest.TestCase):
    def test_retry_success_replaces_error_without_duplicate(self):
        class FakeJudge:
            VALID_ORIGINAL_LABELS = GemmaJudgeV4.VALID_ORIGINAL_LABELS

            def __init__(self, *args):
                self.session = SimpleNamespace(close=lambda: None)

            def judge(self, item):
                return {"success": True, "evaluation": normalize(evidence())}

        with tempfile.TemporaryDirectory() as directory:
            inp, out = Path(directory) / "input.json", Path(directory) / "output.json"
            main_v4.save_json([sample()], inp)
            main_v4.save_json([{
                **main_v4.base_output(sample(), "chunk1__qa1", "error"),
                "error_type": "judge_error", "error": "previous failure",
            }], out)
            args = SimpleNamespace(input=str(inp), output=str(out), limit=None,
                                   no_resume=False, workers=1)
            with patch.object(main_v4, "parse_args", return_value=args), \
                 patch.object(main_v4, "GemmaJudgeV4", FakeJudge):
                main_v4.main(retry_errors_only=True)
            rows = main_v4.load_json(out)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "success")
            self.assertNotIn("retry_attempted", rows[0])

    def test_errors_are_retried_only_by_separate_runner_once(self):
        calls = []

        class FakeJudge:
            VALID_ORIGINAL_LABELS = GemmaJudgeV4.VALID_ORIGINAL_LABELS

            def __init__(self, *args):
                self.session = SimpleNamespace(close=lambda: None)

            def judge(self, item):
                calls.append(item["answer"])
                return {"success": False, "error": "model output invalid"}

        with tempfile.TemporaryDirectory() as directory:
            inp, out = Path(directory) / "input.json", Path(directory) / "output.json"
            main_v4.save_json([sample()], inp)
            args = SimpleNamespace(input=str(inp), output=str(out), limit=None,
                                   no_resume=False, workers=1)
            with patch.object(main_v4, "parse_args", return_value=args), \
                 patch.object(main_v4, "GemmaJudgeV4", FakeJudge):
                main_v4.main()
                self.assertEqual(len(calls), 1)
                main_v4.main()  # Main không tự retry lỗi cũ.
                self.assertEqual(len(calls), 1)
                main_v4.main(retry_errors_only=True)
                self.assertEqual(len(calls), 2)
                self.assertTrue(main_v4.load_json(out)[0]["retry_attempted"])
                main_v4.main(retry_errors_only=True)
                self.assertEqual(len(calls), 2)

    def test_same_chunk_distinct_candidates(self):
        a, b = sample(), sample()
        b["answer"] = "Khác"
        self.assertEqual(
            main_v4.build_sample_ids([a, b]),
            ["chunk1__qa1", "chunk1__qa2"],
        )
        a["id"], b["id"] = "first", "second"
        self.assertEqual(main_v4.build_sample_ids([a, b]), ["first", "second"])

    def test_single_qa_still_has_explicit_qa_suffix(self):
        self.assertEqual(main_v4.build_sample_ids([sample()]), ["chunk1__qa1"])

    def test_duplicate_explicit_id_rejected(self):
        a, b = sample(), sample()
        a["candidate_id"] = b["candidate_id"] = "same"
        with self.assertRaisesRegex(ValueError, "bị trùng"):
            main_v4.build_sample_ids([a, b])

    def test_valid_resume_and_reject_old_or_modified(self):
        row = {"qa_id": "chunk1__qa1", "status": "success"}
        main_v4.validate_resume([row], ["chunk1__qa1"])
        with self.assertRaises(ValueError):
            main_v4.validate_resume([row, row], ["chunk1__qa1"])
        row["qa_id"] = "unknown"
        with self.assertRaises(ValueError):
            main_v4.validate_resume([row], ["chunk1__qa1"])

    def test_compact_output_order_and_fields(self):
        item = main_v4.base_output(sample(), "chunk1__qa1", "success")
        item["evaluation"] = main_v4.compact_evaluation(normalize(evidence()))
        self.assertEqual(list(item)[:3], ["qa_id", "status", "chunk_id"])
        self.assertLess(list(item).index("chunk_id"), list(item).index("context"))
        self.assertNotIn("content", item)
        self.assertNotIn("label", item)
        self.assertEqual(item["original_label"], "mixed_hallucination")
        self.assertNotIn("proposed_label", item["evaluation"])
        self.assertNotIn("unsupported_spans", item["evaluation"])

    def test_legacy_resume_matches_each_qa_by_content(self):
        first, second = sample(), sample()
        second["question"] = "Ai tài trợ?"
        second["answer"] = "Do X tài trợ."
        samples = [first, second]
        identifiers = main_v4.build_sample_ids(samples)
        old = [
            {"_status": "success", "chunk_id": second["chunk_id"],
             "question": second["question"], "answer": second["answer"],
             "evaluation": normalize(evidence())},
            {"_status": "success", "chunk_id": first["chunk_id"],
             "question": first["question"], "answer": first["answer"],
             "evaluation": normalize(evidence())},
        ]
        converted = main_v4.canonicalize_results(old, samples, identifiers)
        self.assertEqual(
            [item["qa_id"] for item in converted],
            ["chunk1__qa2", "chunk1__qa1"],
        )
        self.assertTrue(all(item["status"] == "success" for item in converted))

    def test_bad_input(self):
        for item in (None, [], {**sample(), "content": None}, {**sample(), "answer": ""}):
            with self.assertRaises(ValueError):
                main_v4.normalize(item)

    def test_checkpoint_on_interrupt_and_resume_without_api(self):
        # Lượt 2 bị Ctrl+C: kết quả lượt 1 vẫn lưu dù checkpoint_every=5.
        class FakeJudge:
            VALID_ORIGINAL_LABELS = GemmaJudgeV4.VALID_ORIGINAL_LABELS

            def __init__(self, *args):
                self.session = SimpleNamespace(close=lambda: None)
                self.calls = 0

            def judge(self, item):
                self.calls += 1
                if self.calls == 2:
                    raise KeyboardInterrupt
                return {"success": True, "evaluation": normalize(evidence())}

        with tempfile.TemporaryDirectory() as directory:
            inp, out = Path(directory) / "input.json", Path(directory) / "output.json"
            main_v4.save_json([sample(), {**sample(), "answer": "Khác"}], inp)
            args = SimpleNamespace(input=str(inp), output=str(out), limit=None, no_resume=False)
            with patch.object(main_v4, "parse_args", return_value=args), patch.object(main_v4, "GemmaJudgeV4", FakeJudge):
                with self.assertRaises(KeyboardInterrupt):
                    main_v4.main()
            self.assertEqual(len(main_v4.load_json(out)), 1)
            args.limit = 1
            with patch.object(main_v4, "parse_args", return_value=args), patch.object(main_v4, "GemmaJudgeV4") as factory:
                main_v4.main()
                factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()

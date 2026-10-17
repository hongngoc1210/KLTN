"""Kiểm tra luồng OpenAI Batch offline."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main_judge2_v4_openai_batch import collect, request_line, submit


def sample(qa_id):
    return {"qa_id": qa_id, "chunk_id": "c", "doc_id": "d", "domain": "test",
            "context": "Thủ đô là Hà Nội.", "question": "Thủ đô là gì?", "answer": "Hà Nội.",
            "original_label": "no_hallucination"}


class OpenAIBatchTests(unittest.TestCase):
    def test_request_has_batch_schema_and_no_label(self):
        line = request_line(sample("qa1"), "system", "{context} {question} {answer}")
        self.assertEqual(line["method"], "POST")
        self.assertEqual(line["url"], "/v1/chat/completions")
        self.assertEqual(line["custom_id"], "qa1")
        self.assertNotIn("original_label", json.dumps(line))

    def test_submit_excludes_both_providers_and_live(self):
        rows = [sample(name) for name in ("router", "live", "mini", "new")]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            router = root / "router.json"
            live = root / "live.json"
            state_path = root / "state.json"
            router.write_text(json.dumps({"batches": [{"qa_ids": ["router"]}]}))
            live.write_text(json.dumps([{"qa_id": "live"}]))
            state = {"fingerprint": "fp", "model": "model",
                     "batches": [{"id": "older", "qa_ids": ["mini"]}]}
            def api(method, path, **kwargs):
                if path == "/files":
                    lines = kwargs["files"]["file"][1].decode().splitlines()
                    self.assertEqual([json.loads(line)["custom_id"] for line in lines], ["new"])
                    return {"id": "file1"}
                self.assertEqual(kwargs["json_body"]["input_file_id"], "file1")
                return {"id": "batch1"}
            with patch("main_judge2_v4_openai_batch.api_request", side_effect=api):
                submit(rows, state, state_path, "system", "{answer}",
                       openrouter_state_path=router, live_output_path=live,
                       prompt_fingerprint="new-fp")
            self.assertEqual(state["batches"][-1]["qa_ids"], ["new"])
            self.assertEqual(state["batches"][-1]["id"], "batch1")
            self.assertEqual(state["batches"][-1]["model"], "gpt-5.6-luna")

    def test_collect_joins_output_and_error_files(self):
        rows = [sample("ok"), sample("failed")]
        state = {"batches": [{"id": "batch1", "qa_ids": ["ok", "failed"]}]}
        evaluation = {"question_status": "answerable", "is_evaluable": True,
                      "contradicted_claims": [], "unsupported_claims": [],
                      "proposed_label": "no_hallucination", "alternative_label": None,
                      "self_contradiction": False, "uncertainty_factors": [],
                      "confidence": 0.97, "short_reason": "Đáp án khớp ngữ cảnh."}
        success = {"custom_id": "ok", "response": {"status_code": 200,
                   "body": {"choices": [{"message": {"content": json.dumps(evaluation)}}]}}}
        failure = {"custom_id": "failed", "error": {"message": "rate limit"}}
        def api(method, path, **kwargs):
            if path == "/batches/batch1":
                return {"status": "completed", "output_file_id": "output", "error_file_id": "errors"}
            return json.dumps(success if "output" in path else failure) + "\n"
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output.json"
            with patch("main_judge2_v4_openai_batch.api_request", side_effect=api):
                collect(rows, state, output)
            saved = json.loads(output.read_text())
            self.assertEqual([item["status"] for item in saved], ["success", "error"])
            self.assertEqual(saved[0]["evaluation"]["actual_label"], "no_hallucination")


if __name__ == "__main__":
    unittest.main()

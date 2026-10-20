import copy
import json
from pathlib import Path
import tempfile
import unittest

from sum_models.src.configuration import default_config, matrix
from sum_models.src.core import DIAGNOSES, LABELS, digest, file_hash, input_fields, write_json, write_jsonl
from sum_models.src.data import convert_source, prepare, load_dataset, task_rows
from sum_models.src.features import WindowEncoder, windows
from sum_models.src.metrics import evaluate, matches
from sum_models.prompts.templates import messages, select_demos
from sum_models.src.spans import canonical_spans, decode_bio, parse_prediction


def span(text, quote):
    start = text.index(quote)
    return dict(start=start, end=start + len(quote), quote=quote)


def source(identifier="a", split="train", label="mixed"):
    context, response = "Hà Nội có 10 trường.", "Hà Nội có 20 trường và 3 sân bay."
    def atom(name, relation, diagnosis, quote, evidence, purpose):
        return dict(claim_id=name, evidence_span=evidence,
                    evidence_scoring_eligible=relation == "contradicted", evidence_purpose=purpose,
                    evidence_status="valid" if relation == "contradicted" else "none",
                    annotation=dict(claim_verification_label=relation, diagnosis_type=diagnosis,
                                    hallucinated_spans=[span(response, quote)] if quote else []))
    atoms = [atom("e1", "contradicted", "numerical_distortion", "20 trường", [span(context, "10 trường")], "contradiction"),
             atom("e2", "unsupported", "unsupported_addition", "3 sân bay", [span(context, "Hà Nội")], "repair_generalization")]
    if label == "intrinsic":
        atoms = atoms[:1]
    elif label == "extrinsic":
        atoms = atoms[1:]
    elif label == "no_hallucination":
        atoms = [atom("s1", "supported", "N/A", None, [], "none")]
    return dict(candidate_id=identifier, document_id="doc-" + identifier, split={"split": split},
                domain="education", context=context, question="Nêu số trường.",
                hallucinated_response=response, response_label=label, diagnosis_repair=atoms)


def taxonomy():
    return {"labels": {"diagnosis_type": {d: d for d in DIAGNOSES + ["N/A"]},
                       "claim_verification_label": {v: v for v in ["supported", "contradicted", "unsupported"]}}}


class TinyTokenizer:
    cls_token_id, sep_token_id, bos_token_id, eos_token_id = 1, 2, 1, 2
    def convert_tokens_to_ids(self, token):
        return 3 if token == "<hallu_span>" else 4
    def __call__(self, text, **kwargs):
        return dict(input_ids=[5 + ord(c) % 100 for c in text], offset_mapping=[(i, i + 1) for i in range(len(text))])
    def encode(self, text, **kwargs):
        return self(text)["input_ids"]


class ContractTests(unittest.TestCase):
    def test_honorific_merges_into_existing_entity_diagnosis(self):
        row = source(label="intrinsic")
        row["diagnosis_repair"][0]["annotation"]["diagnosis_type"] = "entity_substitution"
        second = copy.deepcopy(row["diagnosis_repair"][0])
        second["claim_id"] = "honorific-atom"
        second["annotation"]["diagnosis_type"] = "honorific_referent_confusion"
        row["diagnosis_repair"].append(second)
        original_taxonomy = taxonomy()
        original_taxonomy["labels"]["diagnosis_type"]["honorific_referent_confusion"] = "honorific_referent_confusion"
        converted, _ = convert_source(row, original_taxonomy)
        self.assertEqual(len(DIAGNOSES), 10)
        self.assertEqual(converted["regions"][0]["diagnosis_labels"], ["entity_substitution"])
        self.assertEqual([a["diagnosis"] for a in converted["source_atoms"]],
                         ["entity_substitution", "entity_substitution"])
        self.assertEqual(converted["label"], "intrinsic")
        self.assertNotIn("honorific_", json.dumps(converted))
        self.assertNotIn("honorific_", json.dumps(messages(task_rows([converted], "diagnosis")[0], "diagnosis")))

    def test_thirty_configurations(self):
        configs = matrix()
        self.assertEqual(len(configs), 30)
        self.assertEqual([sum(c["tier"] == t for c in configs) for t in [1, 2, 3]], [10, 10, 10])
        self.assertFalse(any(c["tier"] == 2 and c["model"] == "phobert" for c in configs))

    def test_union_and_multilabel(self):
        row = source()
        second = copy.deepcopy(row["diagnosis_repair"][0])
        second["claim_id"] = "e3"
        second["annotation"]["diagnosis_type"] = "circumstance_error"
        row["diagnosis_repair"].append(second)
        converted, _ = convert_source(row, taxonomy())
        self.assertEqual(len(converted["regions"]), 2)
        self.assertEqual(converted["regions"][0]["diagnosis_labels"], ["circumstance_error", "numerical_distortion"])
        self.assertEqual(converted["regions"][1]["evidence"], [])
        self.assertTrue(converted["source_atoms"][1]["auxiliary_repair_evidence"])

    def test_no_target_leakage(self):
        row, _ = convert_source(source(), taxonomy())
        item = task_rows([row], "diagnosis")[0]
        item.update(gold_answer="SECRET", rationale="SECRET", target={"diagnosis_labels": ["SECRET"]}, evidence="SECRET")
        self.assertNotIn("SECRET", json.dumps(messages(item, "diagnosis")))
        self.assertEqual(set(input_fields(item, "diagnosis")), {"context", "question", "response", "span"})

    def test_offsets_repeated_vietnamese_markdown(self):
        row = {"response": "**Hà Nội** và Hà Nội", "context": "Nguồn: Hà Nội"}
        value = {"spans": [{"start": 14, "end": 20, "quote": "Hà Nội", "evidence": []}]}
        self.assertEqual(parse_prediction(json.dumps(value), "localization", row), value)
        value["spans"][0]["start"] = 0
        with self.assertRaises(ValueError):
            parse_prediction(json.dumps(value), "localization", row)

    def test_strict_json(self):
        for text in ['{"label":"intrinsic","label":"mixed"}', '```json\n{"label":"mixed"}\n```', '{"label":NaN}']:
            with self.assertRaises(ValueError):
                parse_prediction(text, "classification", {})

    def test_bio_gap_and_touching(self):
        text = "abcdef"
        offsets = [(i, i + 1) for i in range(6)]
        self.assertEqual(decode_bio([2, 2, 0, 2, 1, 2], offsets, text),
                         [dict(start=0, end=2, quote="ab"), dict(start=3, end=4, quote="d"), dict(start=4, end=6, quote="ef")])
        self.assertEqual(len(canonical_spans([span(text, "ab"), span(text, "cd")], text)), 2)

    def test_windows_cover_every_original_token(self):
        c = dict(default_config(2, "xlmr_base", "finetune"), max_length=32, answer_window_tokens=8, question_tokens=16)
        row, _ = convert_source(source(), taxonomy())
        item = task_rows([row], "localization")[0]
        f = WindowEncoder(TinyTokenizer(), c, "localization").prepare(item)
        self.assertEqual(set(i for b in f["bags"] for i in b["indices"]), set(range(len(row["response"]))))
        self.assertTrue(all(len(b["input_ids"]) <= 32 for b in f["bags"]))
        self.assertEqual(len(f["labels"]), len(row["response"]))
        self.assertGreater(len(f["bags"]), 1)
        self.assertEqual(windows(3, 8, .25), [(0, 3)])

    def test_missing_classification_counts_as_error(self):
        rows = [dict(id=str(i), document_id=str(i), context="", response="", target={"label": label}) for i, label in enumerate(LABELS)]
        preds = {str(i): {"status": "ok", "output": rows[i]["target"]} for i in range(3)}
        r = evaluate(rows, preds, "classification", bootstrap=20)
        self.assertEqual(r["accuracy"], .75)
        self.assertEqual(r["macro_f1"], .75)
        self.assertEqual(r["confusion_matrix"][3][4], 1)

    def test_multilabel_and_absent_class(self):
        rows = [dict(id="1", document_id="1", context="", response="", target={"diagnosis_labels": ["numerical_distortion", "circumstance_error"]})]
        r = evaluate(rows, {"1": {"status": "ok", "output": {"diagnosis_labels": ["numerical_distortion"]}}}, "diagnosis")
        self.assertEqual(r["macro_f1"], .5)
        self.assertEqual(r["exact_set_accuracy"], 0)
        self.assertAlmostEqual(r["micro"]["f1"], 2/3)
        self.assertIn("macro_f1_all_10", r)
        self.assertNotIn("macro_f1_all_11", r)

    def test_span_matching_is_one_to_one(self):
        gold = [dict(start=0, end=4), dict(start=4, end=8)]
        pred = [dict(start=0, end=8)]
        self.assertEqual(len(matches(gold, pred)), 1)

    def test_evidence_is_linked_and_null_is_scored(self):
        r, _ = convert_source(source(), taxonomy())
        row = task_rows([r], "localization")[0]
        output = copy.deepcopy(row["target"])
        output["spans"][0]["evidence"], output["spans"][1]["evidence"] = output["spans"][1]["evidence"], output["spans"][0]["evidence"]
        score = evaluate([row], {row["id"]: {"status": "ok", "output": output}}, "localization")
        self.assertEqual(score["exact"]["f1"], 1)
        self.assertEqual(score["evidence_char"]["f1"], 0)
        self.assertEqual(score["null_evidence"]["accuracy"], 0)

    def test_missing_negative_prediction_not_correct(self):
        row = dict(id="x", document_id="doc", context="", response="abc", target={"spans": []})
        r = evaluate([row], {}, "localization")
        self.assertEqual(r["clean_response_accuracy"]["accuracy"], 0)

    def test_failed_diagnosis_cannot_hide_a_false_positive_span(self):
        row = dict(id="x", document_id="doc", context="", response="abc", target={"spans": []})
        prediction = {"status": "ok", "output": {"spans": [{"start": 0, "end": 1, "quote": "a", "evidence": [], "diagnosis_labels": []}]}}
        r = evaluate([row], {"x": prediction}, "pipeline")
        self.assertEqual(r["joint_exact"]["predicted"], 1)

    def test_few_shot_train_only_distinct_documents(self):
        rows = []
        for i, label in enumerate(LABELS):
            r, _ = convert_source(source(str(i), label=label), taxonomy())
            rows.extend(task_rows([r], "classification"))
        rows.append(dict(rows[0], id="test", split="test", context=""))
        demos = select_demos(rows, "classification")
        self.assertEqual({r["target"]["label"] for r in demos}, set(LABELS))
        self.assertTrue(all(r["split"] == "train" for r in demos))

    def test_prepare_hashes_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = source()
            write_jsonl(root / "source/dataset.jsonl", [raw])
            write_json(root / "source/taxonomy.json", taxonomy())
            write_jsonl(root / "responses.jsonl", [dict(candidate_id=raw["candidate_id"], context=raw["context"], question=raw["question"], response=raw["hallucinated_response"])])
            def manifest():
                write_json(root / "manifest.json", dict(release_id="fixture", audit={"quality_certified": False}, files={name: file_hash(root / name) for name in ["source/dataset.jsonl", "source/taxonomy.json", "responses.jsonl"]}))
            manifest()
            c = {"_root": str(root), "source_manifest": "manifest.json", "output_dir": "prepared"}
            result = prepare(c)
            self.assertEqual(result["counts"]["regions"], 2)
            self.assertEqual(len(load_dataset(root / "prepared/manifest.json")[1]), 1)
            write_jsonl(root / "source/dataset.jsonl", [raw, source("b", "test")])
            manifest()
            c["output_dir"] = "leaked"
            with self.assertRaisesRegex(ValueError, "leakage"):
                prepare(c)


if __name__ == "__main__":
    unittest.main()

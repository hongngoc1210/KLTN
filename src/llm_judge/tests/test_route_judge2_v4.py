"""Kiểm tra offline quy luật route và ID khi nhiều QA chung chunk."""

import unittest

from scripts.judge_pipeline.route_judge2_v4 import split_results


def row(qa_id, **updates):
    evaluation = {
        "label_match": True,
        "alternative_label": None,
        "uncertainty_factors": [],
        "confidence": 1.0,
    }
    evaluation.update(updates)
    return {
        "qa_id": qa_id, "status": "success", "chunk_id": "same_chunk",
        "context": "Đoạn văn.", "question": "Câu hỏi?", "answer": qa_id,
        "original_label": "no_hallucination", "evaluation": evaluation,
    }


class RoutingTests(unittest.TestCase):
    def test_split_and_stable_qa_ids(self):
        rows = [
            row("same_chunk__qa1"),
            row("same_chunk__qa2", label_match=False),
            row("same_chunk__qa3", confidence=0.95),
            row("same_chunk__qa4", uncertainty_factors=["ambiguous_question"]),
            row("same_chunk__qa5", alternative_label="extrinsic_hallucination"),
            {"qa_id": "same_chunk__qa6", "status": "error"},
        ]
        accepted, pending, errors, reasons = split_results(rows)
        self.assertEqual([x["qa_id"] for x in accepted], ["same_chunk__qa1"])
        self.assertEqual(len(pending), 4)
        self.assertEqual(errors, 1)
        self.assertEqual(sum(reasons.values()), 4)
        self.assertEqual([x["qa_id"] for x in pending],
                         ["same_chunk__qa2", "same_chunk__qa3",
                          "same_chunk__qa4", "same_chunk__qa5"])
        self.assertEqual(list(pending[0]),
                         ["qa_id", "chunk_id", "doc_id", "domain",
                          "context", "question", "answer", "original_label"])
        self.assertEqual(pending[0]["original_label"], "no_hallucination")

    def test_duplicate_qa_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "bị trùng"):
            split_results([row("same_chunk__qa1"), row("same_chunk__qa1")])


if __name__ == "__main__":
    unittest.main()

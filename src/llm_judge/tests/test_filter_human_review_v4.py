import unittest

from scripts.human_review.filter_human_review_v4 import (
    build_original_annotations, preserve_completed_reviews, split_samples,
)


def sample(qa_id, original="no_hallucination"):
    return {"qa_id": qa_id, "chunk_id": "c", "doc_id": "d", "domain": "x",
            "context": "ctx", "question": "q", "answer": "a", "original_label": original}


def judged(qa_id, label, confidence, uncertainty=None):
    return {"qa_id": qa_id, "status": "success", "model": "m", "evaluation": {
        "actual_label": label, "confidence": confidence, "question_status": "answerable",
        "alternative_label": None, "uncertainty_factors": uncertainty or [],
        "contradicted_claims": [], "unsupported_claims": [],
        "self_contradiction": False, "short_reason": "reason",
    }}


class FilterHumanReviewTests(unittest.TestCase):
    def test_preserves_completed_review_across_groups(self):
        new = [[{"qa_id": "a", "final_label": "no_hallucination"}], []]
        old = [{"qa_id": "a", "human_completed": True, "human_decision": "drop",
                "human_label": None, "human_contradicted_claims": [],
                "human_unsupported_claims": [], "human_note": "drop", "final_label": None}]
        self.assertEqual(preserve_completed_reviews(new, old), 1)
        self.assertEqual(new[0][0]["human_decision"], "drop")
        self.assertIsNone(new[0][0]["final_label"])

    def test_original_hallucination_spans(self):
        samples = [sample("g", "extrinsic_hallucination"), sample("n")]
        generated = [{"content": "ctx", "question": "q", "hallu_answer": "a sai",
                      "target_label": "extrinsic_hallucination", "hallu_span": "sai"}]
        samples[0]["answer"] = "a sai"
        annotations = build_original_annotations(samples, generated, [])
        self.assertEqual(annotations["g"], {
            "source_model": "deepseek", "hallucination_spans": ["sai"],
        })
        self.assertEqual(annotations["n"]["hallucination_spans"], [])

    def test_all_three_agree_ignores_confidence(self):
        samples = [sample("all")]
        judge1 = [judged("all", "no_hallucination", .80, ["ambiguous_question"])]
        judge2 = [judged("all", "no_hallucination", .82, ["ambiguous_question"])]
        auto_pass, human = split_samples(samples, judge1, judge2)
        self.assertEqual(len(auto_pass), 1)
        self.assertEqual(human, [])
        self.assertEqual(auto_pass[0]["selection_reasons"], ["all_three_labels_agree"])

    def test_rules(self):
        samples = [sample("consensus", "intrinsic_hallucination"), sample("original_j2"),
                   sample("conflict")]
        judge1 = [judged("consensus", "extrinsic_hallucination", .99),
                  judged("original_j2", "extrinsic_hallucination", .90),
                  judged("conflict", "intrinsic_hallucination", .99)]
        judge2 = [judged("consensus", "extrinsic_hallucination", .97),
                  judged("original_j2", "no_hallucination", .97),
                  judged("conflict", "no_hallucination", .97)]
        auto_pass, human = split_samples(samples, judge1, judge2)
        self.assertEqual([row["qa_id"] for row in auto_pass], ["consensus", "original_j2"])
        self.assertEqual([row["qa_id"] for row in human], ["conflict"])
        self.assertIsNone(human[0]["final_label"])
        self.assertEqual(auto_pass[0]["final_label"], "extrinsic_hallucination")


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.human_review.import_human_reviews_v4 import import_reviews


def row(qa_id, note):
    return {"qa_id": qa_id, "content": "c", "question": "q", "answer": "a",
            "original_label": "no_hallucination", "human_label": None,
            "human_note": note, "final_label": None}


class ImportHumanReviewsTests(unittest.TestCase):
    def test_label_and_drop(self):
        outputs = {"human_review": [row("a", "")], "auto_pass": [row("b", "")]}
        reviewed = [row("a", "extrinsic hallucination"), row("b", "drop")]
        counts = import_reviews(reviewed, outputs)
        self.assertEqual(outputs["human_review"][0]["human_label"], "extrinsic_hallucination")
        self.assertEqual(outputs["human_review"][0]["final_label"], "extrinsic_hallucination")
        self.assertEqual(outputs["auto_pass"][0]["human_decision"], "drop")
        self.assertTrue(outputs["auto_pass"][0]["human_completed"])
        self.assertEqual(sum(counts.values()), 2)


if __name__ == "__main__":
    unittest.main()

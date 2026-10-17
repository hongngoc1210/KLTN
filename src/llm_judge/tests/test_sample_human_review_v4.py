import unittest

from scripts.human_review.sample_human_review_v4 import balanced_priority_sample, low_language_quality, tail_sample


class SampleHumanReviewTests(unittest.TestCase):
    def test_low_internal_consistency_is_excluded(self):
        bad = {"qa_id": "bad", "judge1": {"scores": {
            "internal_consistency": 2, "clarity": 5, "fluency": 5,
        }}}
        good = {"qa_id": "good", "judge1": {"scores": {
            "internal_consistency": 5, "clarity": 5, "fluency": 5,
        }}}
        self.assertTrue(low_language_quality(bad))
        self.assertEqual([row["qa_id"] for row in tail_sample([bad, good], 1)], ["good"])

    def test_tail_returns_last_uncompleted(self):
        rows = [
            {"qa_id": "a"}, {"qa_id": "done", "human_completed": True},
            {"qa_id": "b"}, {"qa_id": "c"},
        ]
        self.assertEqual([row["qa_id"] for row in tail_sample(rows, 2)], ["b", "c"])

    def test_excludes_completed_and_prioritizes(self):
        rows = [
            {"qa_id": "done", "human_completed": True, "review_priority": 1, "domain": "a"},
            {"qa_id": "p1a", "review_priority": 1, "domain": "a"},
            {"qa_id": "p1b", "review_priority": 1, "domain": "b"},
            {"qa_id": "p2", "review_priority": 2, "domain": "a"},
        ]
        selected = balanced_priority_sample(rows, 3, seed=42)
        self.assertEqual({row["qa_id"] for row in selected}, {"p1a", "p1b", "p2"})
        self.assertEqual([row["review_priority"] for row in selected], [1, 1, 2])


if __name__ == "__main__":
    unittest.main()

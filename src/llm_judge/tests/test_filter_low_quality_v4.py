import unittest

from scripts.dataset.filter_low_quality_v4 import excluded_ids, filter_rows, low_quality


def gemma(qa_id, consistency=5, clarity=5, fluency=5):
    return {"qa_id": qa_id, "status": "success", "evaluation": {"scores": {
        "internal_consistency": consistency, "clarity": clarity, "fluency": fluency,
    }}}


class FilterLowQualityTests(unittest.TestCase):
    def test_rule_and_synchronized_filter(self):
        rows = [gemma("ok"), gemma("bad_i", consistency=2), gemma("bad_c", clarity=3)]
        excluded, reasons = excluded_ids(rows)
        self.assertEqual(excluded, {"bad_i", "bad_c"})
        self.assertEqual(reasons["internal_consistency"], 1)
        self.assertTrue(low_quality(rows[2]["evaluation"]["scores"]))
        other = [{"qa_id": "ok"}, {"qa_id": "bad_i"}]
        self.assertEqual(filter_rows(other, excluded, "other"), [{"qa_id": "ok"}])


if __name__ == "__main__":
    unittest.main()

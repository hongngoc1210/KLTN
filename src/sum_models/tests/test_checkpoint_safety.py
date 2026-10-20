"""Fail closed when checkpoint pointers are stale, absent or move backwards."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sum_models.src.core import digest, file_hash, write_json, read_json
from sum_models.src.training import point, prune


class CheckpointSafetyTests(unittest.TestCase):
    def checkpoint(self, root, step):
        folder = root / "checkpoints" / f"step-{step:07d}-fixture"
        state = dict(step=step, identity=dict(run_id="fixture", task="classification", smoke_steps=0))
        write_json(folder / "state.json", state)
        write_json(folder / "manifest.json", dict(files={"state.json": file_hash(folder / "state.json")},
                                                 state_hash=digest(state)))
        return folder

    def test_stale_latest_cannot_delete_newer_checkpoint_or_any_older_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = self.checkpoint(root, 200)
            current = self.checkpoint(root, 300)
            newer = self.checkpoint(root, 1300)
            point(root, "latest.json", current)
            with self.assertRaisesRegex(RuntimeError, "cleanup refused"):
                prune(root)
            self.assertTrue(all(p.exists() for p in (older, current, newer)))

    def test_latest_cannot_move_backwards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = self.checkpoint(root, 300)
            newer = self.checkpoint(root, 1300)
            point(root, "latest.json", newer)
            with self.assertRaisesRegex(RuntimeError, "backwards"):
                point(root, "latest.json", older)
            self.assertEqual(read_json(root / "latest.json")["checkpoint"], newer.relative_to(root).as_posix())

    def test_pointer_write_not_visible_stops_before_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = self.checkpoint(root, 300)
            newer = self.checkpoint(root, 400)
            point(root, "latest.json", older)
            with patch("sum_models.src.training.write_json"):
                with self.assertRaisesRegex(RuntimeError, "readback differs"):
                    point(root, "latest.json", newer)
            self.assertTrue(older.exists())
            self.assertTrue(newer.exists())

    def test_missing_latest_never_deletes_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self.checkpoint(root, 1300)
            prune(root)
            self.assertTrue(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()

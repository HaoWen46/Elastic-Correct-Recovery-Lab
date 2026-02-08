from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ecrl.ckpt.atomic_writer import read_latest, write_atomic_checkpoint


class TestAtomicWriter(unittest.TestCase):
    def test_writes_checkpoint_and_latest_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = Path(td)
            payload = {"tensor": torch.tensor([1, 2, 3])}

            latest = write_atomic_checkpoint(
                payload=payload,
                checkpoint_dir=ckpt_dir,
                global_step=7,
                epoch=1,
                world_size_at_save=2,
            )

            ckpt_path = ckpt_dir / "step_00000007.pt"
            self.assertTrue(ckpt_path.exists())
            self.assertEqual(latest["path"], str(ckpt_path))
            self.assertEqual(latest["global_step"], 7)
            self.assertEqual(latest["epoch"], 1)
            self.assertEqual(latest["world_size_at_save"], 2)

            restored = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            self.assertTrue(torch.equal(restored["tensor"], payload["tensor"]))

            latest_from_disk = read_latest(ckpt_dir)
            self.assertIsNotNone(latest_from_disk)
            self.assertEqual(latest_from_disk["global_step"], 7)
            self.assertEqual(Path(latest_from_disk["path"]), ckpt_path)

            temp_files = [p for p in ckpt_dir.iterdir() if p.name.startswith(".step_00000007.pt.")]
            self.assertEqual(temp_files, [])

    def test_latest_pointer_moves_forward(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = Path(td)
            write_atomic_checkpoint(
                payload={"tensor": torch.tensor([1])},
                checkpoint_dir=ckpt_dir,
                global_step=10,
                epoch=2,
                world_size_at_save=4,
            )
            write_atomic_checkpoint(
                payload={"tensor": torch.tensor([2])},
                checkpoint_dir=ckpt_dir,
                global_step=11,
                epoch=2,
                world_size_at_save=4,
            )

            latest = read_latest(ckpt_dir)
            self.assertIsNotNone(latest)
            self.assertEqual(latest["global_step"], 11)
            self.assertEqual(Path(latest["path"]).name, "step_00000011.pt")
            self.assertTrue((ckpt_dir / "step_00000010.pt").exists())
            self.assertTrue((ckpt_dir / "step_00000011.pt").exists())


if __name__ == "__main__":
    unittest.main()

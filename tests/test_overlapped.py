from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ecrl.ckpt.atomic_writer import read_latest
from ecrl.ckpt.overlapped import OverlappedPeriodicCheckpointer


class TestOverlappedCheckpointer(unittest.TestCase):
    def test_flush_wait_drains_inflight_and_persists(self) -> None:
        events = []

        def metrics_logger(record: dict) -> None:
            events.append(record)

        with tempfile.TemporaryDirectory() as td:
            checkpointer = OverlappedPeriodicCheckpointer(
                every_k_steps=1,
                max_inflight=8,
                checkpoint_dir=td,
                rank=0,
                world_size=1,
                metrics_logger=metrics_logger,
            )
            try:
                for step in (1, 2, 3):
                    checkpointer.maybe_checkpoint(
                        global_step=step,
                        epoch=0,
                        capture_fn=lambda s=step: {"step": torch.tensor([s], dtype=torch.int64)},
                    )

                # Explicit flush is required at failure boundaries in overlapped mode.
                checkpointer.flush(wait=True)
                self.assertEqual(checkpointer._inflight, {})

                for step in (1, 2, 3):
                    self.assertTrue((Path(td) / f"step_{step:08d}.pt").exists())

                latest = read_latest(td)
                self.assertIsNotNone(latest)
                self.assertEqual(latest["global_step"], 3)

                completion_events = [e for e in events if e.get("event") == "checkpoint_write_complete"]
                self.assertGreaterEqual(len(completion_events), 1)
            finally:
                checkpointer.close(wait=True)

    def test_nonzero_rank_flush_is_noop(self) -> None:
        checkpointer = OverlappedPeriodicCheckpointer(
            every_k_steps=1,
            max_inflight=2,
            checkpoint_dir="/tmp/unused-overlapped-rank1",
            rank=1,
            world_size=2,
        )
        checkpointer.flush(wait=True)
        checkpointer.close(wait=True)


if __name__ == "__main__":
    unittest.main()

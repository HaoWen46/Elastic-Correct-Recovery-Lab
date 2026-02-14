from __future__ import annotations

import signal
import tempfile
import unittest
from pathlib import Path

from ecrl.orchestration.supervisor import _checkpoint_snapshot, _decode_return_code, _infer_failure_hint


class TestSupervisorHelpers(unittest.TestCase):
    def test_decode_return_code_exit(self) -> None:
        out = _decode_return_code(3)
        self.assertEqual(out["kind"], "exit")
        self.assertEqual(out["exit_code"], 3)

    def test_decode_return_code_signal(self) -> None:
        out = _decode_return_code(-signal.SIGHUP)
        self.assertEqual(out["kind"], "signal")
        self.assertEqual(out["signal"], signal.SIGHUP)
        self.assertEqual(out["signal_name"], "SIGHUP")

    def test_checkpoint_snapshot_counts_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = Path(td)
            ckpt_path = ckpt_dir / "step_00000050.pt"
            ckpt_path.write_bytes(b"checkpoint")
            latest = {"path": str(ckpt_path), "global_step": 50}

            snap = _checkpoint_snapshot(ckpt_dir, latest)

            self.assertTrue(snap["latest_exists"])
            self.assertEqual(snap["latest_path"], str(ckpt_path))
            self.assertTrue(snap["latest_path_exists"])
            self.assertEqual(snap["checkpoint_file_count"], 1)

    def test_infer_failure_hint_for_sighup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "attempt.log"
            log_path.write_text(
                "SignalException: Process 123 got signal: 1\n"
                "Received 1 death signal, shutting down workers\n",
                encoding="utf-8",
            )
            hint = _infer_failure_hint(1, attempt_log_path=log_path)
            self.assertEqual(hint, "torchrun_received_sighup")


if __name__ == "__main__":
    unittest.main()

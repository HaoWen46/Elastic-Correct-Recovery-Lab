from __future__ import annotations

import unittest

import numpy as np
import torch

from ecrl.statepack.rng import capture_rng_state, restore_rng_state


class TestStatepackRng(unittest.TestCase):
    def test_restore_accepts_non_byte_tensor_cpu_state(self) -> None:
        state = capture_rng_state()
        # Simulate a loaded checkpoint where dtype/device changed by map_location.
        state["torch_cpu"] = state["torch_cpu"].to(dtype=torch.int64)
        restore_rng_state(state)
        self.assertEqual(torch.get_rng_state().dtype, torch.uint8)

    def test_restore_accepts_numpy_array_cpu_state(self) -> None:
        state = capture_rng_state()
        state["torch_cpu"] = np.asarray(state["torch_cpu"].cpu().numpy(), dtype=np.uint8)
        restore_rng_state(state)
        self.assertEqual(torch.get_rng_state().dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()

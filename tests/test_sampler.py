from __future__ import annotations

import unittest

from ecrl.sampler.resumable_sampler import ResumableGlobalBatchSampler


class TestResumableGlobalBatchSampler(unittest.TestCase):
    def test_deterministic_iteration_for_same_seed_epoch(self) -> None:
        s1 = ResumableGlobalBatchSampler(
            dataset_size=100,
            global_batch=20,
            world_size=4,
            rank=1,
            seed=1234,
            epoch=3,
            cursor_step=0,
        )
        s2 = ResumableGlobalBatchSampler(
            dataset_size=100,
            global_batch=20,
            world_size=4,
            rank=1,
            seed=1234,
            epoch=3,
            cursor_step=0,
        )

        ids1 = list(iter(s1))
        ids2 = list(iter(s2))
        self.assertEqual(ids1, ids2)
        self.assertEqual(len(ids1), s1.steps_per_epoch * s1.local_batch)

    def test_state_roundtrip_preserves_remaining_stream(self) -> None:
        s1 = ResumableGlobalBatchSampler(
            dataset_size=120,
            global_batch=24,
            world_size=2,
            rank=0,
            seed=7,
            epoch=2,
            cursor_step=2,
        )
        state = s1.state_dict()

        s2 = ResumableGlobalBatchSampler(
            dataset_size=120,
            global_batch=24,
            world_size=2,
            rank=0,
            seed=0,
            epoch=0,
            cursor_step=0,
        )
        s2.load_state_dict(state)

        self.assertEqual(state, s2.state_dict())
        self.assertEqual(list(iter(s1)), list(iter(s2)))

        old_cursor = s2.cursor_step
        s2.advance_step()
        self.assertEqual(s2.cursor_step, old_cursor + 1)

    def test_elastic_rank_partition_matches_global_window(self) -> None:
        sampler = ResumableGlobalBatchSampler(
            dataset_size=103,
            global_batch=20,
            world_size=1,
            rank=0,
            seed=99,
            epoch=1,
            cursor_step=0,
        )

        for step in range(sampler.steps_per_epoch):
            full = sampler.local_ids_for(epoch=1, cursor_step=step, rank=0, world_size=1)

            from_four = []
            for rank in range(4):
                from_four.extend(
                    sampler.local_ids_for(epoch=1, cursor_step=step, rank=rank, world_size=4)
                )

            from_two = []
            for rank in range(2):
                from_two.extend(
                    sampler.local_ids_for(epoch=1, cursor_step=step, rank=rank, world_size=2)
                )

            self.assertEqual(full, from_four)
            self.assertEqual(full, from_two)
            self.assertEqual(len(full), sampler.global_batch)
            self.assertEqual(len(set(full)), sampler.global_batch)

    def test_rejects_non_divisible_batch(self) -> None:
        with self.assertRaises(ValueError):
            ResumableGlobalBatchSampler(
                dataset_size=100,
                global_batch=10,
                world_size=3,
                rank=0,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()

## Limitations and Next Steps (Draft)
- Current matrix uses one seed per suite; variance estimates are not robust.
- Checkpoint sensitivity is not covered in this run (`k=50`, `max_inflight=4` fixed).
- Recommended next run under quota constraints:
  1. Keep same matrix and rerun with `SEEDS_CSV=1337,2027`.
  2. Add `CHECKPOINT_EVERY_CSV=50,100` for CIFAR-100 suites first.
  3. Keep the same `MATRIX_PREFIX` and resume mode to avoid recomputing completed suites.

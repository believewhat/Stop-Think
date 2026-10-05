# Publication validation (2026-10-05)

- 15 dependency-light prompt, completion-label, packaging, and FSDP-prefix tests passed.
- 22 MedQA reward/parser regression tests passed.
- All published Python files passed AST parsing.
- The DAPO Bash example passed shell syntax checking.
- Documentation relative links resolved; Git whitespace checks passed.
- No model weights, prepared trajectories, raw evaluation outputs, server
  credentials, or private deployment paths were added.

These checks were run on a CPU-only Windows host. Full PyTorch/FSDP/vLLM
integration and GPU execution were **not** run for this publication. The
additional CUDA-dependent regressions are supplied, not claimed as passing
here. Run the one-update smoke configuration in the target Linux environment.

An unused historical math reward test in the source snapshot referenced the
removed `_well_formed_stop_format` API; it is not part of this MedQA release.
A separate historical math-verifier test also failed on the Windows host's
multiprocessing timeout backend. Neither was represented as a passing MedQA
test, and the existing released math pipeline was not changed.

Portable adaptations: extracted SFT data helpers, renamed local imports,
parameterized model/data/runtime/output paths, removed server supervision,
and added the explicit isolated-runtime builder. The MedQA runtime overlay's
training/reward/classifier implementations are otherwise retained from the
current development source snapshot. Checkpoint and hardware limitations are
documented in the training README.

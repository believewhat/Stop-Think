# Third-party notices

Detaching this repository from a GitHub fork network changes repository
metadata, not code authorship or licensing.

- **vLLM**: vendored and modified under `vllm/` and
  `training/medqa/runtime_overlay/vllm/`; upstream
  <https://github.com/vllm-project/vllm>, Apache License 2.0.
- **veRL**: vendored and modified under `EarlyStop/verl/` and
  `training/medqa/runtime_overlay/verl/`; upstream
  <https://github.com/volcengine/verl>, Apache License 2.0.
  Original copyright notices remain in the source files.
- **AdaptThink / EarlyStop**: the existing MIT notice is retained in
  [EarlyStop/LICENSE](EarlyStop/LICENSE), copyright 2025 THU-KEG.
- The LoRA SFT example is **SWITCH-inspired**, not a reproduction of every
  component of that method. It uses PEFT, Transformers, and PyTorch APIs.

The Apache 2.0 text is included at
[licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
Retain each component's original notices when redistributing it. Model and
dataset weights are not included; their own upstream licenses still apply.

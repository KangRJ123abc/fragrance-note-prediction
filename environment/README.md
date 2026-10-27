# Separate historical environments

The `*_pip_freeze.txt` files are historical version snapshots with local build
paths removed. They are not a universal pip installation recipe. CUDA builds,
Python versions, wheel sources, and GPU compatibility must be selected explicitly.

| Purpose | Key frozen versions |
| --- | --- |
| Training | Unsloth/Unsloth Zoo 2026.7.2; torch 2.10.0+cu130; Transformers 5.5.0; TRL 0.24.0; xformers 0.0.34 |
| Domain and application inference | vLLM 0.24.0; torch 2.11.0; Transformers 5.13.0; Python 3.12 |
| General benchmark client | EvalScope 1.4.2 in a separate environment |

`requirements-*.txt` list principal dependencies for each role. Install an
appropriate CUDA-enabled PyTorch build first where needed. ModelScope is only
needed for downloads, not for loading an already-downloaded model directory.
The benchmark client starts a vLLM executable from the separate inference
environment, supplied through `--vllm-executable`.

No CUDA/GPU execution has been performed while assembling this release.

# Credits and licences

- **vLLM 0.32.0**, Apache-2.0, https://github.com/vllm-project/vllm. Used unmodified (installed from PyPI). `scripts/slo_scheduler.py`
  subclasses vLLM's `Scheduler` / `AsyncScheduler` through the `--scheduler-cls` extension point, so it is a derivative-style plug-in;
  keep the vLLM licence notice when redistributing it.
- **PyTorch** (BSD-3-Clause), **safetensors** (Apache-2.0), **compressed-tensors** (Apache-2.0, config schema of the INT8 checkpoint),
  **huggingface_hub** (Apache-2.0), **httpx** (BSD-3-Clause), **nvidia-ml-py** (BSD-3-Clause, NVML energy readout in the replay tools).
- **uv** (Apache-2.0 / MIT) for environment creation.
- **Base model**: `CMKL/MANGO1.5-Qwen3.5-9B` by CMKL, built on Qwen3.5-9B (Alibaba Qwen team). Our INT8 weights are a derivative and
  carry the base model's licence (see its model card on Hugging Face).
- Code in this repository written for the hackathon: ThunderPermanent (Eakkachat Sathaphon, Paradee Pimoleakaksorn, Pongsura Pongpisan, Thanakrit Samoena). Licence for our code: Apache License 2.0 (see `LICENSE`, same as vLLM; the licence choice is to be confirmed by the repository owner).

See branches for different implementation on this playground

## Device support

`transformers_playground.py` auto-selects CUDA/ROCm/CPU at runtime. `torch.compile` is enabled on CUDA/ROCm and disabled on CPU; fused AdamW and FlexAttention are disabled on AMD/ROCm and CPU to avoid backend-specific failures, while keeping the standard attention path as a fallback.

When running on CPU, the script uses reduced defaults (`seq_len=64`, `batch_size=16`, `total_steps=200`) and skips warmup to keep runtime reasonable. Perplexity and generation settings are also reduced on CPU to avoid long stalls.

## Runtime UX

Progress reporting uses `tqdm.auto` so it renders in both notebooks and terminals. Live matplotlib updates are only enabled in notebooks.

## AMD/ROCm setup

If ROCm is installed (for example `/opt/rocm` exists) but PyTorch is CUDA-only, the script fails fast with guidance. Install ROCm wheels that match your system ROCm version, for example:

```bash
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/rocm6.3 torch torchvision torchaudio
```

## Weights & Biases logging

`transformers_playground.py` uses W&B if available. Set `WANDB_API_KEY` to enable logging; otherwise it skips `wandb.init`. Initialization uses a 120s timeout to avoid transient startup timeouts.

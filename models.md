# Model Compatibility Log

## Tested Models

### Llama 3.1-8B-Instruct (`meta-llama/Llama-3.1-8B-Instruct`)

**Status: Fully working**

- Key norms are moderate and uniform across all 32 layers (max ~20, no outliers)
- Norm guard keeps 0/32 layers dense — all layers quantize cleanly
- 3-bit Q_mse: key cosine sim 0.983, value cosine sim 0.983 — coherent output
- 4-bit Q_mse: key cosine sim 0.995, value cosine sim 0.995 — near-lossless
- 3-bit Q_prod (QJL keys + quantize decode): key cosine sim 0.920 — works, slightly lower quality
- Memory savings: 80% KV payload reduction at 3-bit

**Long-context results (3-bit Q_mse, chunked attention):**

| Prompt tokens | Dense KV | Packed KV | Peak VRAM overhead | Savings |
|:---:|:---:|:---:|:---:|:---:|
| 5K | 477 MB | 93 MB | 161 MB (vs 499 MB baseline) | 67.6% |
| 36K | 4.77 GB | 932 MB | 999 MB (vs 4.86 GB baseline) | 79.5% |
| 73K | 9.54 GB | 1.86 GB | 1.94 GB (vs 9.71 GB baseline) | 80.1% |

At 73K tokens, TurboQuant saves 7.8 GB of peak VRAM — the difference between
fitting on a 24 GB consumer GPU vs needing 32 GB+.

### Qwen 2.5-7B-Instruct (`Qwen/Qwen2.5-7B-Instruct`)

**Status: Works with norm guard (3 layers kept dense)**

- Qwen 2.5 has pathologically high key norms in specific layers:
  - Layer 0: key_mean_norm = 273.7
  - Layer 1: key_mean_norm = 66.3
  - Layer 27: key_mean_norm = 239.5
  - Normal layers (e.g., Layer 10): key_mean_norm = 16.8
- These extreme norms cause catastrophic Q_prod failure without norm guard
  because QJL logit variance scales with norm² — at norm 239, logit std ~4.5
  swamps typical attention logits of O(10)
- Norm guard automatically detects these layers and keeps them dense (3/28)
- With norm guard: 3-bit Q_prod produces coherent output, 71.5% savings
- Q_mse (without QJL) also works well with norm guard

**Why the paper didn't hit this:** the paper tested on Llama-3.1-8B and
Ministral-7B, which don't have Qwen's "massive activation" pathology.

## Architecture Independence

TurboQuant operates on the KV cache after projection and RoPE — it is
architecture-agnostic in principle. The norm guard handles model-specific
activation pathologies automatically. No per-model code paths are needed;
the same configuration works across model families.

## Known Issues

- **Per-channel outlier splitting makes Q_prod worse**: splitting 128 dims into
  32 outlier + 96 normal concentrates energy in fewer QJL dimensions, increasing
  variance. Recommend `num_outlier_channels=0` for Q_prod.
- **Decode speed tradeoff**: chunked online-softmax attention in pure PyTorch is
  ~2-5x slower than fused CUDA SDPA. Acceptable for VRAM-constrained users;
  will be addressed by Triton kernel.

## TODO

- [ ] **Triton kernel for fused attention from packed data** — reads packed
  indices/signs directly in shared memory, never materializing full K/V. This
  would give both memory AND speed improvements, eliminating the current latency
  penalty from pure-PyTorch chunked attention. The spec is the existing
  `chunked_turboquant_attention()` function in `src/turboquant/runtime/attention.py`.
- [ ] Test on Ministral-7B (paper's other validated model)
- [ ] Test on larger models (70B+ class) where KV cache dominates VRAM
- [ ] Benchmark on LongBench / NIAH at scale for quality validation
- [ ] Adaptive per-layer bit allocation (more bits for high-norm layers instead
  of binary dense/quantized)

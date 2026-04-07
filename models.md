# Tested Models & Algorithm Notes

## Tested Models

### Llama 3.1 8B Instruct
- **Status:** Fully working — all 32 layers quantize cleanly
- **Best config:** 4-bit Q_prod (default)
- Key norms moderate and uniform (~15-21 across all layers)
- Norm guard triggers on 0/32 layers
- 4-bit: cosine sim ~0.995, 74% KV savings
- 3-bit: cosine sim ~0.983, 80% KV savings
- Architecture: 32 layers, 32 query heads, 8 KV heads (GQA), D=128

### Qwen 2.5 7B Instruct
- **Status:** Works with norm guard (3/28 layers kept dense)
- **Best config:** 4-bit Q_mse with dense decode (`--no-qjl --no-quantize-decode`)
- 3 pathological layers with extreme key norms:
  - Layer 0: norm 273.7 — 5 channels hold 50% of energy
  - Layer 27: norm 239.5 — 3 channels hold 50%, max/min ratio 266,774x
  - Layer 1: norm 66.3
- Remaining 25 layers behave normally (norm ~16.8)
- With norm guard: 71.5% KV savings, coherent output
- Architecture: 28 layers, 28 query heads, 4 KV heads (GQA), D=128

### Not yet tested
- **DeepSeek-R1/V3** — MLA architecture may interact differently with TurboQuant
- **Gemma 2/3** — mentioned in TurboQuant blog post, likely well-behaved
- **Ministral-7B** — paper's other validated model, should work cleanly
- **Llama 3.3 70B** — scale test where KV cache dominates VRAM

---

## Why both Q_prod and Q_mse?

The paper presents Q_prod as the primary algorithm. We support both
because model-specific activation patterns make one better than the other.

**Q_prod** (default, `--use-qjl-keys`):
- Keys: (b-1)-bit MSE + 1-bit QJL sign correction
- Logit is an **unbiased** estimate of `<q, k>` — prevents attention drift
- Works perfectly on Llama (moderate, uniform key norms)
- Breaks on Qwen's extreme-norm layers without norm guard

**Q_mse** (`--no-qjl`):
- Keys: all b bits for MSE reconstruction
- Higher cosine similarity (16 vs 8 codebook levels at 4-bit)
- Logit is **biased** — can drift during long generation
- Bias is mitigated by keeping decode tokens dense (`--no-quantize-decode`)
- Safer for models with pathological key norms

### The tradeoff

| | Q_prod | Q_mse |
|--|--------|-------|
| Logit bias | None (unbiased) | Small systematic bias |
| Cosine sim (4-bit) | ~0.975 | ~0.995 |
| Decode tokens | Can re-quantize safely | Should keep dense |
| Extreme norms | Variance blows up | Handles gracefully |
| Best for | Llama, long generation | Qwen, quality-sensitive |

### Why Qwen breaks with Q_prod

Q_prod's QJL variance is `(π/2 - 1) * ||r||^2 / D`. For Qwen layer 27:
- Key norm = 239.5 → residual norm is large
- QJL logit std ≈ 4.5
- Typical non-dominant attention logits are O(10)
- Noise of std 4.5 can flip which keys get attention → softmax corruption

Norm guard solves this by auto-detecting layers where the key norm exceeds
a threshold and keeping them in full precision. On Qwen, layers 0, 1, 27
stay dense; the other 25 layers are quantized normally.

The paper tested on Llama and Ministral, which don't have this pathology.

---

## Kernel status

**Fused Triton kernel — shipped.** Supports both Q_mse and Q_prod.

Decode latency is ~2.25x baseline (fused CUDA SDPA). The gap is
irreducible bit-unpacking ALU + codebook gather that dense attention
doesn't need. Optimization targets (split-K, autotune) aim for ~1.5x.

| Context | Baseline | TurboQuant | VRAM saved |
|---------|----------|------------|------------|
| ~40K | 7.0s | 15.7s | 3.9 GB (74%) |

The latency gap narrows at longer context because the fixed per-position
overhead becomes a smaller fraction of total work.

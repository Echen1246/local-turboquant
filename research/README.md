# TurboQuant — Research & Implementation Details

For installation and usage, see the [root README](../README.md).
For tested models and known issues, see [models.md](../models.md).

---

## Algorithm

### Q_mse (TurboQuant_mse)

Quantizes each KV vector to minimize reconstruction MSE:

```
x ∈ R^d
  → normalize:   u = x / ||x||          (store ||x|| as FP16 scale)
  → rotate:      z = u @ R^T             (fixed random orthogonal R)
  → quantize:    idx = LloydMax(z_j)     (per-coordinate, Beta distribution)
  → dequantize:  z_hat = centers[idx]
  → unrotate:    u_hat = z_hat @ R
  → rescale:     x_hat = u_hat * ||x||
```

The random orthogonal rotation is critical — it decorrelates coordinates
so that each dimension follows a `Beta((d-1)/2, (d-1)/2)` distribution,
making a scalar Lloyd-Max codebook near-optimal per the rate-distortion
bound. Without the rotation, outlier channels (especially in Qwen) would
require per-channel handling.

### Q_prod (TurboQuant_prod)

Extends Q_mse to produce unbiased attention logits:

```
x_hat_mse = Q_mse(x, b-1 bits)          (MSE quantization at b-1 bits)
r = x - x_hat_mse                        (residual)
qjl = sign(r @ S^T)                      (1-bit JL sketch, random Gaussian S)
store: (mse_indices, qjl_signs, ||r||)

logit = <q, x_hat_mse> + sqrt(π/2)/d * ||r|| * <q @ S^T, qjl>
```

**The guarantee:** `E[logit] = <q, x>` — the logit is an unbiased
estimator of the true inner product. This prevents systematic attention
drift during long autoregressive generation, which is the paper's core
contribution.

**Key constraint:** The two-part representation (MSE + QJL) must be
consumed directly in a custom attention kernel. Merging the QJL correction
back into a single vector produces dense noise — the QJL term is optimized
for linear statistics, not vector reconstruction.

### Bit budget

Q_prod uses the SAME total bit budget as Q_mse. At b bits per dimension:

| Method | Keys | Values | Total bits/dim |
|--------|------|--------|----------------|
| Q_mse | b-bit MSE (2^b levels) | b-bit MSE | b |
| Q_prod | (b-1)-bit MSE + 1-bit QJL | b-bit MSE | b |

At 4 bits: Q_prod keys get 8 codebook levels (3-bit MSE) + 1-bit QJL
sign. Same storage as 4-bit Q_mse, but with unbiased logits instead of
optimal reconstruction.

---

## Kernel integration

### Fused Triton attention kernel

The fused kernel (`_tile_attention_kernel` in `triton_kernels.py`)
computes attention directly from compressed storage in a single GPU pass:

```
Grid: (n_tiles, Q_heads)
Per tile (TILE_N=64 positions × D=128 dimensions):

  1. Unpack key MSE indices from bit-packed uint8
  2. Gather key codebook centers
  3. Compute MSE logit: norm_k * <q @ R_k^T, centers[k_idx]>
  4. [Q_prod only] Unpack 1-bit QJL signs
  5. [Q_prod only] Add correction: sqrt(π/2)/D * ||r|| * <q @ S^T, signs>
  6. Online softmax (tile-local max + exp + sum)
  7. Unpack value MSE indices, gather value centers
  8. Weighted value accumulation
  9. Store partial (out, max, sum) for cross-tile reduction
```

The `HAS_QJL` constexpr flag means Q_mse compiles to identical code
as before — zero overhead when QJL is not used.

**Separate codebooks:** Q_prod uses different bit-widths for keys
(b-1 bits) and values (b bits), so the kernel accepts separate
`key_centers_ptr`/`val_centers_ptr` and `KEY_BITS`/`VAL_BITS`.

**Dense buffer merge:** When `quantize_decode=False`, newly generated
tokens stay in a dense buffer. The kernel output is merged with
standard SDPA on the dense tokens via online softmax combination.

### Fallback chain

```
turboquant_attention_forward()
  → fused_attention()          # Triton kernel (GPU, B=1, Sq=1)
  → chunked_turboquant_attention()  # PyTorch chunked (any device/shape)
```

If Triton is unavailable or the batch/sequence shape is unsupported,
the chunked PyTorch path handles it. Both paths operate from compressed
storage — no full KV decompression.

---

## Q_mse vs Q_prod: when to use which

**Q_prod (default, `--use-qjl-keys`):**
- Paper-faithful, unbiased attention logits
- Prevents attention drift in long generation
- Works cleanly on Llama (moderate key norms)
- Slightly lower cosine similarity (8 vs 16 codebook levels at 4-bit)

**Q_mse (`--no-qjl`):**
- Maximum reconstruction fidelity
- Safer for models with extreme key norms (Qwen layers 0, 1, 27)
- Compatible with any attention backend
- Best paired with dense decode buffer (`--no-quantize-decode`)

**Never merge Q_prod back into a stored vector.** The QJL correction
is noise optimized for linear statistics. Using it as a normal KV cache
entry breaks the guarantee.

---

## Expected tradeoffs

### VRAM savings scale with context length

```
Llama 3.1 8B, 4-bit Q_prod, all 32 layers quantized:

Context     Dense KV    Packed KV    Savings    Overhead saved
────────    ────────    ─────────    ───────    ──────────────
  ~8K        655 MB      172 MB       74%         483 MB
 ~40K       5243 MB     1372 MB       74%        3936 MB
 ~73K       9540 MB     2490 MB       74%        7050 MB
```

Savings percentage is constant (~74% at 4-bit). Absolute MB saved
grows linearly with context — the longer the context, the bigger the
win.

### Latency trades off against VRAM

```
Llama 3.1 8B, 4-bit Q_prod, fused Triton kernel:

Context     Baseline    TurboQuant    Ratio
────────    ────────    ──────────    ─────
 ~40K        7.0s        15.7s        2.25x
```

The ~2.25x slowdown comes from ALU work that baseline SDPA doesn't
need: bit-unpacking, codebook lookup, QJL dot product. Future
optimizations (split-K parallelism, autotune, SRAM staging) target
~1.3-1.5x.

**The tradeoff is worth it when VRAM is the bottleneck.** At 73K context,
baseline needs ~25 GB for KV alone. TurboQuant fits it in ~7 GB — the
difference between needing an A100 and fitting on a consumer 24 GB card.

### Latency gap narrows at longer context

The kernel's compute cost is proportional to context length (same as
baseline SDPA). The constant overhead (codebook lookup, bit ops) becomes
a smaller fraction of total work as context grows. At very long contexts,
the ratio approaches ~1.5x with the current kernel.

---

## Measured results

### Reconstruction quality

Llama 3.1 8B, per-layer average across all 32 layers:

```
Bits    Key Cosine Sim    Val Cosine Sim    Key MSE      Val MSE
────    ──────────────    ──────────────    ───────      ───────
  4     0.9954            0.9954            0.0278       0.0009
  3     0.9830            0.9830            (higher)     (higher)
  2     visibly lossy     visibly lossy     —            —
```

### Key norm pathology (Qwen vs Llama)

Why Qwen needs norm guard and Llama doesn't:

```
Model    Layer    Key Mean Norm    50% Energy In    Max/Min Ratio
─────    ─────    ─────────────    ─────────────    ─────────────
Qwen       0     273.7            5 channels       extreme
Qwen      27     239.5            3 channels       266,774x
Qwen      10     16.8             15 channels      normal
Llama     all    ~15-21           distributed      moderate
```

Q_prod variance scales with `||r||^2 / D`. At Qwen layer 27 (norm 239),
logit std ~4.5 swamps typical attention logits of O(10). Norm guard
auto-detects these layers and keeps them dense.

### QwQ-32B NIAH (original research)

3-bit Q_mse, NIAH at 4K/8K/16K/32K tokens, depths 10/50/90%:

```
Variant        Match %    KV Size
───────        ───────    ───────
baseline       100%       3.98 GB
qmse_packed    100%       778 MB    (80.4% savings)
```

---

## Research structure

```
research/
  modal_app.py              NIAH benchmark runner (Modal, QwQ-32B)
  config.py                 Model ID, GPU, revision pins
  benchmarks/
    niah.py                 NIAH protocol (needle generation, scoring)
    paper.py                Paper benchmark specifications
  modeling/
    qwq.py                  QwQ-32B loading helper
  quantization/
    attention_metrics.py    Causal logit MSE measurement
  runtime/
    experiment_log.py       JSONL experiment logging
    kv_artifacts.py         KV tensor extraction
    kv_capture.py           KV cache capture and summarization
```

### Reproduce (QwQ-32B NIAH)

```bash
pip install -e ".[benchmarks,modal,dev]"
modal setup

modal run research/modal_app.py --prefetch-only
modal run research/modal_app.py \
  --niah-grid --variant qmse_packed --qmse-bits 3 \
  --context-lengths 4000,8000,16000,32000 \
  --depth-percents 10,50,90 --run-name niah-packed-b3
```

---

## References

- TurboQuant: https://openreview.net/forum?id=tO3ASKZlok
- QJL: https://arxiv.org/abs/2406.03482
- PolarQuant: https://arxiv.org/abs/2502.02617
- Google Research blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

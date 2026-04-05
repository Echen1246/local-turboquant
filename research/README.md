# TurboQuant Research

Research harness for TurboQuant KV-cache compression. This directory contains
the original QwQ-32B experiments, NIAH benchmark infrastructure, and KV
capture/analysis tools.

For the installable library and setup instructions, see the
[root README](../README.md). For tested models and known issues, see
[models.md](../models.md).

## Algorithm

TurboQuant_mse quantizes each KV vector independently:

```
x ∈ R^d
  → normalize:   u = x / ||x||          (store ||x|| as scale)
  → rotate:      z = u R^T              (fixed random orthogonal R)
  → quantize:    idx = LloydMax(z_j)    (per-coordinate, Beta distribution)
  → dequantize:  z_hat = Centroid(idx)
  → unrotate:    u_hat = z_hat R
  → rescale:     x_hat = u_hat * ||x||
```

The coordinate distribution on a unit sphere in d dimensions is
`Beta((d-1)/2, (d-1)/2)` — this is why a Lloyd-Max codebook trained on
that distribution is near-optimal per the rate-distortion bound.

TurboQuant_prod extends this for unbiased inner products:

```
x_hat_mse = Q_mse(x, b-1 bits)
r = x - x_hat_mse                       (residual)
qjl = sign(S * r)                       (1-bit JL sketch, random Gaussian S)
store: (mse_indices, qjl_signs, ||r||)

logit = <q, x_hat_mse> + sqrt(π/2)/d * ||r|| * <Sq, qjl>
```

This gives `E[logit] = <q, x>` (unbiased). The two-part representation
must be consumed directly in a custom attention kernel — merging the QJL
correction back into a single vector produces garbage.

## What is implemented

- `TurboQuant_mse` with Lloyd-Max codebook for Beta-distributed coordinates
- `TurboQuant_prod` with QJL residual sign sketch
- Packed bit-packed cache with norm storage (`PackedMSELayer`)
- Chunked online-softmax attention from compressed data
- Lazy update mode (new tokens stored directly in packed form)
- Range decoding (decompress only the chunk needed for attention)
- Norm guard (auto-detects high-norm layers and keeps them dense)
- Per-channel outlier-aware mixed precision

## Measured results

All benchmarks on Modal B200, `sdpa` attention backend.

### Reconstruction quality

Llama 3.1-8B-Instruct, per-layer average across all 32 layers:

```
┌──────┬──────────────────┬──────────────────┬──────────────┬──────────────┐
│ Bits │ Key Cosine Sim   │ Val Cosine Sim   │ Key MSE      │ Val MSE      │
├──────┼──────────────────┼──────────────────┼──────────────┼──────────────┤
│  4   │ 0.9954           │ 0.9954           │ 0.0278       │ 0.0009       │
│  3   │ 0.9830           │ 0.9830           │ (higher)     │ (higher)     │
│  2   │ visibly lossy    │ visibly lossy    │ —            │ —            │
└──────┴──────────────────┴──────────────────┴──────────────┴──────────────┘
```

Qwen 2.5-7B-Instruct, 3-bit Q_prod with norm guard (25/28 layers quantized):

```
┌────────────────────────────────────┬─────────┐
│ Metric                             │ Value   │
├────────────────────────────────────┼─────────┤
│ Key cosine sim (quantized layers)  │ 0.920   │
│ Layers quantized                   │ 25/28   │
│ Layers kept dense (norm guard)     │ 3/28    │
│ Dense layers                       │ 0, 1, 27│
│ KV payload savings                 │ 71.5%   │
└────────────────────────────────────┴─────────┘
```

### Long-context memory savings

Llama 3.1-8B-Instruct, 3-bit Q_mse, chunked attention:

```
┌──────────────┬───────────┬───────────┬──────────────┬─────────┐
│ Prompt Tokens│ Dense KV  │ Packed KV │ Peak Overhead│ Savings │
├──────────────┼───────────┼───────────┼──────────────┼─────────┤
│  5,000       │ 477 MB    │ 93 MB     │ 161 MB       │ 67.6%   │
│ 36,000       │ 4.77 GB   │ 932 MB    │ 999 MB       │ 79.5%   │
│ 73,000       │ 9.54 GB   │ 1.86 GB   │ 1.94 GB      │ 80.1%   │
└──────────────┴───────────┴───────────┴──────────────┴─────────┘
```

At 73K tokens, TurboQuant saves 7.8 GB of peak VRAM — the difference
between fitting on a 24 GB consumer GPU vs needing 32 GB+.

### Decode speed tradeoff

Llama 3.1-8B-Instruct, 73K context, 16 new tokens:

```
┌────────────────┬────────────────┬────────────────┐
│                │ Baseline       │ TurboQuant     │
├────────────────┼────────────────┼────────────────┤
│ Decode time    │ 6.0s           │ 29.4s          │
│ Relative speed │ 1.0x           │ ~0.2x          │
└────────────────┴────────────────┴────────────────┘
```

The ~5x slowdown at long context is the pure-PyTorch chunked attention
kernel. A fused Triton kernel reading directly from packed indices in
shared memory would eliminate this. The spec for that kernel is
`chunked_turboquant_attention()` in `src/turboquant/runtime/attention.py`.

### Key norm pathology (Qwen vs Llama)

Channel energy profiling revealed why Qwen needs norm guard:

```
┌───────┬────────┬───────────────┬─────────────────┬──────────────────┐
│ Model │ Layer  │ Key Mean Norm │ 50% Energy In   │ Max/Min Ratio    │
├───────┼────────┼───────────────┼─────────────────┼──────────────────┤
│ Qwen  │  0     │ 273.7         │ 5 channels      │ extreme          │
│ Qwen  │  1     │ 66.3          │ moderate         │ high             │
│ Qwen  │ 10     │ 16.8          │ 15 channels     │ normal           │
│ Qwen  │ 27     │ 239.5         │ 3 channels      │ 266,774x         │
├───────┼────────┼───────────────┼─────────────────┼──────────────────┤
│ Llama │ all    │ ~15-21        │ distributed     │ moderate         │
└───────┴────────┴───────────────┴─────────────────┴──────────────────┘
```

Qwen's pathological layers concentrate >50% of key energy in 3-5 channels.
Q_prod variance scales with norm²: at norm 239, logit std ~4.5 swamps
typical attention logits of O(10). Norm guard solves this by keeping
those layers dense automatically.

### QwQ-32B NIAH results (original research)

QwQ-32B, 3-bit Q_mse, NIAH at 4K/8K/16K/32K tokens, depths 10/50/90%:

```
┌──────────────┬──────────┬──────────┬──────────────┐
│ Variant      │ Match %  │ Dense KV │ Packed KV    │
├──────────────┼──────────┼──────────┼──────────────┤
│ baseline     │ 100%     │ 3.98 GB  │ —            │
│ qmse         │ 100%     │ 3.98 GB  │ —            │
│ qmse_packed  │ 100%     │ —        │ 778 MB       │
└──────────────┴──────────┴──────────┴──────────────┘

Payload reduction: 80.4%
```

## Q_mse vs Q_prod: when to use which

**Use Q_mse when:**
- You want a drop-in cache format
- You are validating quantization quality
- You are integrating with an existing attention kernel
- You want maximum reconstruction fidelity

**Use Q_prod (`--use-qjl-keys`) when:**
- You have the TurboQuant custom attention kernel active
- You want unbiased attention logits (paper-faithful)
- You are willing to accept lower cosine similarity for the
  theoretical guarantee of unbiased inner products

**Never merge Q_prod back into a plain vector.** The QJL correction
is noise optimized for linear statistics. Storing it as a normal KV
cache entry breaks the guarantee and produces garbage.

## Research structure

```
research/
  modal_app.py              NIAH benchmark runner (Modal, QwQ-32B)
  config.py                 Model ID, GPU, revision pins
  sources.py                Paper/benchmark source catalog
  benchmarks/
    niah.py                 NIAH protocol (needle generation, scoring)
    paper.py                Paper benchmark specifications
  modeling/
    qwq.py                  QwQ-32B loading helper
  quantization/
    attention_metrics.py    Causal logit MSE measurement
  runtime/
    experiment_log.py       JSONL experiment logging
    kv_artifacts.py         KV tensor extraction from safetensors
    kv_capture.py           KV cache capture and summarization
    metadata.py             Run naming, timestamps, JSON I/O
    query_capture.py        Query projection capture
```

## Paper references

- TurboQuant: https://openreview.net/forum?id=tO3ASKZlok
- QJL: https://arxiv.org/abs/2406.03482
- PolarQuant: https://arxiv.org/abs/2502.02617
- Google blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

## Reproduce (QwQ-32B NIAH)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[benchmarks,modal,dev]"
modal setup

# Warm cache
modal run research/modal_app.py --prefetch-only

# Baseline NIAH grid
modal run research/modal_app.py \
  --niah-grid \
  --context-lengths 4000,8000,16000,32000 \
  --depth-percents 10,50,90 \
  --variant baseline \
  --run-name niah-baseline

# Packed 3-bit NIAH grid
modal run research/modal_app.py \
  --niah-grid \
  --context-lengths 4000,8000,16000,32000 \
  --depth-percents 10,50,90 \
  --variant qmse_packed \
  --qmse-bits 3 \
  --run-name niah-qmse-packed-b3

# Compare
modal run research/modal_app.py \
  --compare-niah-baseline niah-baseline \
  --compare-niah-candidate niah-qmse-packed-b3
```

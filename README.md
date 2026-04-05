# TurboQuant

A Python library implementing DeepMind's TurboQuant paper (ICLR 2026) for
KV cache compression on HuggingFace Transformers LLMs. Drop-in KV cache
quantization that reduces VRAM usage by 70-80% for long-context inference.

> **Research details:** For the full algorithm breakdown, Q_mse vs Q_prod
> math, benchmark methodology, and QwQ-32B research results, see
> [research/README.md](research/README.md).
>
> **Model support:** For currently tested models, known issues, limitations,
> and work in progress (Triton kernel), see [models.md](models.md).

---

## What it does

TurboQuant compresses the key-value cache during inference so that long
contexts fit in less GPU memory. At 3-bit quantization on Llama 3.1-8B,
a 73K-token context uses 1.9 GB instead of 9.7 GB — an 80% reduction.

The library integrates with HuggingFace Transformers via a custom attention
backend. A chunked online-softmax kernel processes compressed history in
chunks without ever materializing the full decompressed KV cache, so peak
VRAM savings are real, not just storage savings.

### How TurboQuant works

For each KV vector `x`:

1. Normalize to unit sphere, store the original norm separately
2. Apply a fixed random orthogonal rotation
3. Quantize each rotated coordinate with an optimal Lloyd-Max scalar
   quantizer (the coordinate distribution on a random unit sphere is a
   known Beta distribution)
4. Inverse-rotate and rescale to reconstruct

This is the **Q_mse** path — it minimizes vector reconstruction MSE and is
the safe drop-in choice. The **Q_prod** path adds a 1-bit QJL sign sketch
on the residual to get unbiased attention logits, but requires a custom
attention kernel to consume the two-part representation correctly.

---

## Prerequisites

- Python 3.11+
- A CUDA-capable NVIDIA GPU (for real inference)
- A HuggingFace account with access to gated models (e.g. Llama)

## Installation

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install git+https://github.com/Echen1246/local-turboquant.git

# Or for local development
git clone https://github.com/Echen1246/local-turboquant.git
cd local-turboquant
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install git+https://github.com/Echen1246/local-turboquant.git
```

> **Note:** macOS does not have CUDA GPUs. TurboQuant will run on CPU for
> testing purposes, but real inference requires an NVIDIA GPU. Use
> [Modal](#remote-gpu-inference-with-modal) or an SSH-connected Linux
> machine with a GPU.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install git+https://github.com/Echen1246/local-turboquant.git

# Or for local development
git clone https://github.com/Echen1246/local-turboquant.git
cd local-turboquant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### Windows (CMD)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install git+https://github.com/Echen1246/local-turboquant.git
```

## HuggingFace authentication

Many models (Llama, Gemma, etc.) require accepting a license on HuggingFace
before you can download them. After accepting the license on the model's
HuggingFace page:

```bash
# Option 1: environment variable (Linux/macOS)
export HF_TOKEN="hf_your_token_here"

# Option 1: environment variable (Windows PowerShell)
$env:HF_TOKEN = "hf_your_token_here"

# Option 2: pass directly to TurboQuant
turboquant run --model meta-llama/Llama-3.1-8B-Instruct --token hf_your_token_here ...
```

---

## Quick start: activate on any model

The simplest way to use TurboQuant — activate it on your existing model
and use `model.generate()` normally:

```python
import turboquant
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype="auto",
)

# Activate TurboQuant — one line, that's it
turboquant.activate(model, tokenizer, bits=4)
#   TurboQuant activated
#     Model:      llama (32 layers)
#     Bits:       4-bit Q_mse
#     Norm guard: on

# Now use model.generate() exactly as before
inputs = tokenizer("What is KV cache compression?", return_tensors="pt").to("cuda")
output = model.generate(inputs.input_ids, max_new_tokens=256)
# [TurboQuant] 4-bit Q_mse active | call #1
print(tokenizer.decode(output[0], skip_special_tokens=True))

# Check how well it compressed
telemetry = turboquant.last_telemetry(model)
print(f"Savings: {telemetry['payload_savings_percent']:.1f}%")

# Deactivate when done (restores original model.generate)
turboquant.deactivate(model)
# [TurboQuant] Deactivated after 1 calls.
```

Every call to `model.generate()` prints a one-line status confirming
TurboQuant is active. Pass `quiet=True` to suppress it.

## Quick start: session API

For more control (metrics, multiple prompts, compatibility checks):

```python
from turboquant import TurboQuantSession

session = TurboQuantSession.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    variant="qmse_packed",
    bits=4,
    device_map="auto",
    dtype="auto",
)

text = session.generate(
    messages=[{"role": "user", "content": "What is KV cache compression?"}],
    max_new_tokens=256,
)
print(text)

telemetry = session.last_telemetry()
print(f"Dense KV:  {telemetry['dense_kv_bytes'] / 1e9:.2f} GB")
print(f"Packed KV: {telemetry['packed_actual_bytes'] / 1e9:.2f} GB")
print(f"Savings:   {telemetry['payload_savings_percent']:.1f}%")
```

---

## CLI commands

### System detection

```bash
turboquant setup
```

Detects your GPU, VRAM, CUDA version, Python/PyTorch/Transformers versions,
HuggingFace token status, and recommends models and bit widths for your hardware.

### Show library info

```bash
turboquant info
```

```
TurboQuant v0.1.0

Quantization modes:
  qmse_packed    Packed Q_mse cache (default, recommended)
  qmse           Dense reconstructed Q_mse cache
  baseline       No quantization (for comparison)

Supported bit widths: 2, 3, 4

Tested models:
  meta-llama/Llama-3.1-8B-Instruct  fully working, 80% savings at 73K ctx
  Qwen/Qwen2.5-7B-Instruct          works with norm guard (3/28 layers dense)
```

### Check model compatibility

```bash
turboquant inspect --model meta-llama/Llama-3.1-8B-Instruct
```

### Run with telemetry

```bash
# 4-bit quantization with full telemetry output
turboquant run \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --variant qmse_packed \
  --bits 4 \
  --prompt "Explain quantum computing in one paragraph." \
  --show-telemetry

# 3-bit Q_prod (paper-faithful) with metrics
turboquant run \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --variant qmse_packed \
  --bits 3 \
  --use-qjl-keys \
  --quantize-decode \
  --prompt "Explain quantum computing." \
  --show-telemetry --show-metrics

# Full JSON output (for scripting/logging)
turboquant run \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --variant qmse_packed \
  --bits 4 \
  --prompt "Hello" \
  --json
```

### Display telemetry from a saved run

```bash
turboquant run \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --bits 4 \
  --prompt "Hello" \
  --json > run_output.json

turboquant telemetry run_output.json
```

---

## Quantization modes

| Mode | Keys | Values | Decode tokens | Best for |
|------|------|--------|--------------|----------|
| Q_mse (default) | Q_mse | Q_mse | Dense | Maximum quality |
| Q_prod | Q_mse + QJL | Q_mse | Quantized | Paper-faithful, max savings |

### Settings guide

| Setting | Quality | Compression | Use case |
|---------|---------|-------------|----------|
| 4-bit Q_mse | Near-lossless (cosine sim 0.995) | ~75% savings | Production, quality-sensitive |
| 3-bit Q_mse | Very good (cosine sim 0.983) | ~80% savings | Long context, VRAM-constrained |
| 3-bit Q_prod | Good (cosine sim 0.920) | ~80% savings | Maximum savings, decode quant |
| 2-bit Q_mse | Lossy | ~87% savings | Experimental only |

### Key flags

- `--bits 3|4` — quantization bit width (4-bit is near-lossless)
- `--use-qjl-keys` — enable Q_prod for keys (adds QJL sign sketch)
- `--quantize-decode` — quantize decode-phase tokens too
- `--no-norm-guard` — disable automatic dense fallback for high-norm layers

---

## Telemetry reference

When you run with `--show-telemetry`, TurboQuant reports:

| Metric | What it measures |
|--------|-----------------|
| `dense_kv_bytes` | Size of the uncompressed KV cache |
| `packed_actual_bytes` | Size of the compressed KV cache in GPU memory |
| `payload_savings_percent` | Compression ratio (higher = better) |
| `post_cache_setup_allocated_bytes` | GPU VRAM after cache is built |
| `peak_allocated_bytes` | Peak GPU VRAM during generation |
| `generation_seconds` | Wall-clock time for the full generation |
| `quantization_seconds` | Time spent quantizing the prefill cache |

With `--show-metrics` or `--json`, you also get per-layer reconstruction
quality (cosine similarity, MSE) so you can verify quantization fidelity.

---

## Smoke tests

Local (CPU, small model):

```bash
python examples/local_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct --bits 4
```

Modal GPU:

```bash
modal run examples/modal_smoke.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --variant qmse_packed \
  --bits 4
```

Memory benchmark (compares baseline vs TurboQuant VRAM):

```bash
modal run examples/modal_smoke.py --memory-benchmark --prompt-tokens 32768
```

## Remote GPU inference with Modal

If you don't have a local GPU, use [Modal](https://modal.com) for cloud
GPU access:

```bash
pip install turboquant[modal]

modal setup

modal run examples/modal_smoke.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --variant qmse_packed \
  --bits 4

# Memory benchmark (baseline vs TurboQuant)
modal run examples/modal_smoke.py --memory-benchmark --prompt-tokens 32768
```

---

## Repo structure

```
src/turboquant/              # Installable library
  quantization/              #   Core TurboQuant_mse algorithm
  runtime/                   #   Packed cache, attention kernel, generation loop
    attention.py             #   Chunked online-softmax attention from packed data
    packed_qmse_cache.py     #   Bit-packed KV cache with range decoding
    generation.py            #   Greedy decode with prefill-then-quantize flow
  adapters/                  #   HuggingFace Transformers integration
  cli.py                     #   CLI entry point (turboquant command)
  telemetry.py               #   Telemetry summarization
examples/                    # Smoke tests, benchmarks, visualization
  local_smoke.py             #   Local CPU/GPU test
  modal_smoke.py             #   Modal cloud GPU test with memory benchmark
  benchmark_suite.py         #   Configuration sweep benchmark runner
  visualize.py               #   Chart generation from benchmark data
research/                    # Original QwQ-32B research harness
  modal_app.py               #   NIAH benchmark runner on Modal
  benchmarks/                #   NIAH protocol, paper benchmark specs
  See research/README.md for full details
models.md                    # Tested models, known issues, Triton TODO
```

---

## Background: Q_mse vs Q_prod

### Q_mse (TurboQuant_mse)

The MSE-optimized quantization path:

1. Normalize `x` to unit norm, store `||x||` separately
2. Apply random orthogonal rotation to decorrelate coordinates
3. Quantize each coordinate with Lloyd-Max scalar quantizer (the
   coordinate distribution on a unit sphere is Beta-distributed)
4. Dequantize by inverse rotation and norm rescaling

**Guarantees:** near-optimal MSE rate, small residual, high cosine similarity.
This is the right tool for drop-in KV cache compression where reconstructed
vectors feed directly into standard attention.

### Q_prod (TurboQuant_prod)

The inner-product-optimized path:

1. Run Q_mse at `b-1` bits
2. Compute residual `r = x - x_hat_mse`
3. Apply QJL to residual: `qjl = sign(S * r)` (random Gaussian matrix)
4. Store `(mse_indices, qjl_signs, ||r||)`

**Guarantees:** for any query `q`, `E[<q, x_hat_prod>] = <q, x>` (unbiased
logits). **Does NOT guarantee** low vector MSE, high cosine similarity, or
stability after softmax.

**Why this matters:** naively merging the QJL correction back into a stored
vector and using it as a normal KV cache entry produces garbage — the QJL
term is dense noise optimized for linear statistics, not vector reconstruction.
The correct path consumes the two parts separately in a custom attention kernel,
which is what TurboQuant implements via `chunked_turboquant_attention()`.

### When to use which

- **Q_mse:** drop-in compression, maximum quality, works with any attention backend
- **Q_prod (--use-qjl-keys):** paper-faithful path for keys, requires the
  TurboQuant custom attention kernel (automatically activated)

---

## Paper references

- TurboQuant paper: https://openreview.net/forum?id=tO3ASKZlok
- QJL paper: https://arxiv.org/abs/2406.03482
- PolarQuant paper: https://arxiv.org/abs/2502.02617
- Google Research blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

---

## Troubleshooting

**`OSError: You are trying to access a gated repo`**
Accept the model's license on HuggingFace and set `HF_TOKEN`.
See [HuggingFace authentication](#huggingface-authentication).

**`torch.cuda.OutOfMemoryError`**
Try a smaller model, reduce `--max-new-tokens`, or use a lower bit width
(e.g. `--bits 3` instead of `--bits 4`).

**Slow generation**
The current attention kernel is pure PyTorch. A Triton kernel for fused
attention from packed data is on the roadmap (see [models.md](models.md)).
TurboQuant trades decode speed for VRAM — it's designed for fitting longer
contexts, not faster generation.

**`ModuleNotFoundError: No module named 'turboquant'`**
Make sure your virtual environment is activated and TurboQuant is installed.
If running from the repo, use `pip install -e .` from the repo root.

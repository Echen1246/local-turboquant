<img width="246" height="50" alt="Screenshot 2026-04-06 at 8 53 35 PM" src="https://github.com/user-attachments/assets/78510409-d458-40a9-829e-6f59d3c7b2b6" />

KV cache compression for HuggingFace Transformers. Drop-in ~74% VRAM
reduction for long-context inference, based on Google Research's TurboQuant
(ICLR 2026).

> [research/README.md](research/README.md) — algorithm, kernel design, benchmarks
> | [models.md](models.md) — tested models, known issues
> | [remote_setup.md](remote_setup.md) — Vast.ai, Lambda, Modal, Colab

---

## Install

```bash
pip install git+https://github.com/Echen1246/local-turboquant.git
```

Requires Python 3.11+ and an NVIDIA GPU with CUDA.

---

## Usage

TurboQuant hooks into your existing Transformers code. One line to
activate, then use `model.generate()` normally — no changes to your
prompts, tokenizer, or generation config.

```python
import turboquant
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load any HuggingFace model as usual
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype="auto",
)

# Activate — all generate() calls now use compressed KV cache
turboquant.activate(model, tokenizer)
```

That's it. Everything after `activate()` works exactly like normal
Transformers:

```python
inputs = tokenizer("What is KV cache compression?", return_tensors="pt").to("cuda")
output = model.generate(inputs.input_ids, max_new_tokens=1024)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# Works with chat templates, system prompts, generation configs — anything
messages = [{"role": "user", "content": "Explain attention mechanisms."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
output = model.generate(inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# Check compression stats anytime
print(turboquant.last_telemetry(model))
# → {'payload_savings_percent': 73.8, 'dense_kv_bytes': ..., 'packed_actual_bytes': ...}
```

To go back to normal:

```python
turboquant.deactivate(model)
# model.generate() is now the original, uncompressed version
```

### What activate() does

- Saves the original `model.generate` method
- Replaces it with a wrapper that compresses the KV cache using
  TurboQuant's fused Triton kernel
- Your `max_new_tokens`, `temperature`, `attention_mask`, etc. all
  pass through unchanged — TurboQuant imposes no limits
- `deactivate()` restores the original method
<img width="1190" height="173" alt="image" src="https://github.com/user-attachments/assets/336b803a-6e6c-48fd-9725-7e352e71ef94" />

### Options

```python
turboquant.activate(model, tokenizer)              # 4-bit Q_prod (default)
turboquant.activate(model, tokenizer, bits=3)       # 3-bit, more compression
turboquant.activate(model, tokenizer, use_qjl_keys=False)  # Q_mse instead of Q_prod
```

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `bits` | 4 | Quantization width (4 = near-lossless, 3 = very good) |
| `use_qjl_keys` | True | Q_prod (paper-accurate unbiased logits) |
| `quantize_decode` | True | Re-quantize generated tokens into packed cache |
| `norm_guard` | True | Auto-keep high-norm layers dense (needed for Qwen) |

---

## CLI

For quick testing without writing Python:

```bash
# One-shot prompt (defaults to Llama 3.1 8B, 4-bit Q_prod)
turboquant run --prompt "Explain KV cache compression."

# Interactive REPL
turboquant attach

# System info (GPU, VRAM, Triton, recommendations)
turboquant setup

# Different model or bit width
turboquant run --model Qwen/Qwen2.5-7B-Instruct --bits 3 --prompt "Hello"
```

---

## What gets installed

Only the `turboquant` Python package (`src/turboquant/`). The `research/`,
`examples/`, docs, and notebooks are NOT included in the pip install —
they live in the repo for reference but don't ship with the library.

```
pip package:
  turboquant/
    quantization/     Core TurboQuant_mse algorithm
    runtime/          Packed cache, Triton kernel, attention, generation
    adapters/         HuggingFace Transformers integration
    cli.py            CLI entry point
    api.py            Public API (activate, deactivate, etc.)

repo only (not installed):
  research/           Algorithm research, QwQ-32B benchmarks
  examples/           Smoke tests, Colab notebooks, Modal scripts
  models.md           Tested models and known issues
  remote_setup.md     Cloud GPU setup guides
```

---

## Tested models

| Model | Status | Savings |
|-------|--------|---------|
| Llama 3.1 8B Instruct | Fully working | 74% |
| Qwen 2.5 7B Instruct | Works with norm guard | 71.5% |

See [models.md](models.md) for details.

---

## HuggingFace token

Gated models (Llama, Gemma) need a token:

```bash
export HF_TOKEN="hf_your_token_here"
```

---

## Troubleshooting

**Gated repo error** — Accept the license on HuggingFace and set `HF_TOKEN`.

**CUDA OOM** — Try `bits=3` or a shorter context.

**Slow generation** — Expected: ~2x baseline latency at long context.
TurboQuant trades decode speed for VRAM savings.

---

## Paper

- [TurboQuant (ICLR 2026)](https://openreview.net/forum?id=tO3ASKZlok)
- [QJL](https://arxiv.org/abs/2406.03482)
- [Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

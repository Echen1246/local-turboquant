from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import modal

from turboquant import TurboQuantSession

app = modal.App("turboquant-smoke")
_local_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
function_secrets = (
    [
        modal.Secret.from_dict(
            {
                "HF_TOKEN": _local_hf_token,
                "HUGGINGFACE_HUB_TOKEN": _local_hf_token,
            }
        )
    ]
    if _local_hf_token
    else []
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==1.13.0",
        "huggingface_hub[hf_transfer]==1.6.0",
        "numpy==2.4.3",
        "safetensors==0.7.0",
        "scipy==1.17.1",
        "torch==2.10.0",
        "transformers==5.3.0",
    )
    .env({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/root/src"})
    .add_local_dir(
        str(Path(__file__).resolve().parents[1] / "src"),
        remote_path="/root/src",
        ignore=["__pycache__", "*.pyc"],
    )
)


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=8192,
    timeout=30 * 60,
    secrets=function_secrets,
)
def run_smoke(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    variant: str = "qmse_packed",
    bits: int = 3,
    prompt: str = "Explain KV cache compression in one short paragraph.",
    max_new_tokens: int = 128,
) -> dict[str, object]:
    session = TurboQuantSession.from_pretrained(
        model,
        variant=variant,
        bits=bits,
        dtype="auto",
        device_map="auto",
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    text = session.generate(prompt=prompt, max_new_tokens=max_new_tokens)
    return {
        "model": model,
        "variant": variant,
        "bits": bits if variant != "baseline" else None,
        "text": text,
        "telemetry": session.last_telemetry(),
        "compatibility": session.compatibility_report(),
    }


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    variant: str = "qmse_packed",
    bits: int = 3,
    prompt: str = "Explain KV cache compression in one short paragraph.",
    max_new_tokens: int = 128,
) -> None:
    result = run_smoke.remote(
        model=model,
        variant=variant,
        bits=bits,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    print(result)

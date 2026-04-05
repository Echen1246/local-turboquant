from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import modal

from turboquant import TurboQuantSession

app = modal.App("turboquant-smoke")
HF_CACHE_DIR = "/vol/hf-cache"
hf_cache_volume = modal.Volume.from_name("tq-local-hf-cache", create_if_missing=True)
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
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": HF_CACHE_DIR,
            "PYTHONPATH": "/root/src",
        }
    )
    .add_local_dir(
        str(Path(__file__).resolve().parents[1] / "src"),
        remote_path="/root/src",
        ignore=["__pycache__", "*.pyc"],
    )
)


@app.cls(
    image=image,
    gpu="B200",
    cpu=4,
    memory=16384,
    timeout=30 * 60,
    secrets=function_secrets,
    volumes={HF_CACHE_DIR: hf_cache_volume},
    scaledown_window=20 * 60,
    min_containers=1,
)
class SmokeRunner:
    @modal.enter()
    def load(self) -> None:
        self._sessions: dict[tuple, TurboQuantSession] = {}

    def _session_for(
        self,
        model: str,
        variant: str,
        bits: int,
        num_outlier_channels: int = 0,
        outlier_extra_bits: int = 1,
        use_qjl_keys: bool = False,
        quantize_decode: bool = False,
        norm_guard: bool = True,
    ) -> TurboQuantSession:
        key = (model, variant, bits, num_outlier_channels, outlier_extra_bits,
               use_qjl_keys, quantize_decode, norm_guard)
        session = self._sessions.get(key)
        if session is None:
            session = TurboQuantSession.from_pretrained(
                model,
                variant=variant,
                bits=bits,
                dtype="auto",
                device_map="auto",
                token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
                cache_dir=HF_CACHE_DIR,
                num_outlier_channels=num_outlier_channels,
                outlier_extra_bits=outlier_extra_bits,
                use_qjl_keys=use_qjl_keys,
                quantize_decode=quantize_decode,
                norm_guard=norm_guard,
            )
            self._sessions[key] = session
            hf_cache_volume.commit()
        return session

    @modal.method()
    def run(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        variant: str = "qmse_packed",
        bits: int = 3,
        prompt: str = "Explain KV cache compression in one short paragraph.",
        max_new_tokens: int = 128,
        num_outlier_channels: int = 0,
        outlier_extra_bits: int = 1,
        use_qjl_keys: bool = False,
        quantize_decode: bool = False,
        norm_guard: bool = True,
    ) -> dict[str, object]:
        session = self._session_for(
            model=model,
            variant=variant,
            bits=bits,
            num_outlier_channels=num_outlier_channels,
            outlier_extra_bits=outlier_extra_bits,
            use_qjl_keys=use_qjl_keys,
            quantize_decode=quantize_decode,
            norm_guard=norm_guard,
        )
        output = session.generate(prompt=prompt, max_new_tokens=max_new_tokens, return_output=True)
        text = output.text
        effective_bits = bits
        if variant != "baseline" and num_outlier_channels > 0:
            cfg = session.model.config
            head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
            normal_channels = head_dim - num_outlier_channels
            effective_bits = (
                num_outlier_channels * (bits + outlier_extra_bits) + normal_channels * bits
            ) / head_dim
        recon = output.metrics.reconstruction_quality
        recon_summary = None
        if recon:
            quantized = [r for r in recon if not r.get("dense", False)]
            dense_layers = [r["layer"] for r in recon if r.get("dense", False)]
            src = quantized if quantized else recon
            avg_key_cos = sum(r["key_cosine_sim"] for r in src) / len(src)
            avg_val_cos = sum(r["val_cosine_sim"] for r in src) / len(src)
            avg_key_mse = sum(r["key_mse"] for r in src) / len(src)
            avg_val_mse = sum(r["val_mse"] for r in src) / len(src)
            worst_key_mse_layer = max(src, key=lambda r: r["key_mse"])
            max_key_norm = max(r["key_mean_norm"] for r in recon)
            recon_summary = {
                "layers_quantized": len(quantized),
                "layers_dense": len(dense_layers),
                "dense_layer_ids": dense_layers,
                "avg_key_cosine_sim": round(avg_key_cos, 6),
                "avg_val_cosine_sim": round(avg_val_cos, 6),
                "avg_key_mse": round(avg_key_mse, 6),
                "avg_val_mse": round(avg_val_mse, 6),
                "max_key_mean_norm": round(max_key_norm, 2),
                "worst_key_mse_layer": worst_key_mse_layer,
            }

        return {
            "model": model,
            "variant": variant,
            "bits": bits if variant != "baseline" else None,
            "num_outlier_channels": num_outlier_channels if variant != "baseline" else None,
            "outlier_extra_bits": outlier_extra_bits if variant != "baseline" else None,
            "use_qjl_keys": use_qjl_keys if variant != "baseline" else None,
            "quantize_decode": quantize_decode if variant != "baseline" else None,
            "norm_guard": norm_guard if variant != "baseline" else None,
            "effective_bits": round(effective_bits, 3) if variant != "baseline" else None,
            "text": text,
            "reconstruction_quality": recon_summary,
            "telemetry": session.last_telemetry(),
            "compatibility": session.compatibility_report(),
        }


    @modal.method()
    def profile_channels(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        prompt: str = "Explain KV cache compression in one short paragraph.",
    ) -> dict[str, object]:
        """Profile per-channel key energy to determine if extreme norms are concentrated or distributed."""
        import torch

        session = self._session_for(model=model, variant="baseline", bits=3)
        inputs = session.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = session.model(**inputs, use_cache=True)
            past_kv = outputs.past_key_values

        layer_profiles = []
        for layer_idx, (key_states, val_states, *_) in enumerate(past_kv):
            k = key_states.float()
            channel_energy = k.pow(2).sum(dim=(0, 1, 2))  # [head_dim]
            total_energy = channel_energy.sum().item()

            sorted_energy, sorted_idx = channel_energy.sort(descending=True)
            cumulative = sorted_energy.cumsum(0) / total_energy

            n_50 = int((cumulative < 0.5).sum().item()) + 1
            n_90 = int((cumulative < 0.9).sum().item()) + 1
            n_99 = int((cumulative < 0.99).sum().item()) + 1

            top5_idx = sorted_idx[:5].tolist()
            top5_pct = [round(sorted_energy[i].item() / total_energy * 100, 2) for i in range(5)]

            mean_norm = k.norm(dim=-1).mean().item()
            max_channel = sorted_energy[0].item()
            min_channel = sorted_energy[-1].item()
            ratio = max_channel / min_channel if min_channel > 0 else float("inf")

            layer_profiles.append({
                "layer": layer_idx,
                "key_mean_norm": round(mean_norm, 2),
                "channels_for_50pct_energy": n_50,
                "channels_for_90pct_energy": n_90,
                "channels_for_99pct_energy": n_99,
                "top5_channels": top5_idx,
                "top5_pct_of_total": top5_pct,
                "max_min_channel_ratio": round(ratio, 1),
            })

        flagged = [lp for lp in layer_profiles if lp["key_mean_norm"] > 50]
        return {
            "model": model,
            "head_dim": int(key_states.shape[-1]),
            "num_layers": len(layer_profiles),
            "flagged_layers": flagged,
            "normal_layer_example": layer_profiles[10] if len(layer_profiles) > 10 else layer_profiles[-1],
        }


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    variant: str = "qmse_packed",
    bits: int = 3,
    prompt: str = "Explain KV cache compression in one short paragraph.",
    max_new_tokens: int = 128,
    num_outlier_channels: int = 0,
    outlier_extra_bits: int = 1,
    use_qjl_keys: bool = False,
    quantize_decode: bool = False,
    norm_guard: bool = True,
    profile_channels: bool = False,
) -> None:
    import json

    if profile_channels:
        result = SmokeRunner().profile_channels.remote(model=model, prompt=prompt)
    else:
        result = SmokeRunner().run.remote(
            model=model,
            variant=variant,
            bits=bits,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_outlier_channels=num_outlier_channels,
            outlier_extra_bits=outlier_extra_bits,
            use_qjl_keys=use_qjl_keys,
            quantize_decode=quantize_decode,
            norm_guard=norm_guard,
        )
    print(json.dumps(result, indent=2, default=str))

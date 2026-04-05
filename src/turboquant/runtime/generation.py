from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

from turboquant.quantization.turboquant_mse import quantize_past_key_values_mse
from turboquant.runtime.memory_accounting import (
    gpu_current_memory_bytes,
    gpu_peak_memory_bytes,
    past_key_values_memory_breakdown,
    turboquant_mse_packed_bytes,
)
from turboquant.runtime.packed_qmse_cache import (
    build_packed_mse_cache,
    packed_cache_storage_breakdown,
    verify_packed_reconstruction,
)


@dataclass(frozen=True)
class GenerationMetrics:
    variant: str
    qmse_bits: int | None
    quantization_seconds: float
    generation_seconds: float
    prompt_tokens: int
    completion_tokens: int
    prefill_cache: dict[str, Any]
    turboquant_mse_packed_estimate: dict[str, Any] | None
    turboquant_mse_packed_actual: dict[str, Any] | None
    post_cache_setup_gpu_memory: dict[str, Any] | None
    gpu_peak_memory: dict[str, Any] | None
    reconstruction_quality: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    metrics: GenerationMetrics


def validate_generation_variant(variant: str) -> None:
    if variant not in {"baseline", "qmse", "qmse_packed"}:
        raise ValueError(
            f"Unsupported generation variant={variant!r}. "
            "Choose from ['baseline', 'qmse', 'qmse_packed']."
        )


def eos_token_ids(model, tokenizer) -> set[int]:
    eos_token_id = getattr(model.generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return {int(item) for item in eos_token_id}


def greedy_decode_with_prefill_cache(
    *,
    model,
    tokenizer,
    inputs,
    max_new_tokens: int,
    variant: str,
    qmse_bits: int,
    rotation_seed: int = 0,
    num_outlier_channels: int = 0,
    outlier_extra_bits: int = 1,
    use_qjl_keys: bool = False,
    quantize_decode: bool = False,
    norm_guard: bool = True,
) -> GenerationOutput:
    import torch

    validate_generation_variant(variant)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    decode_started = time.monotonic()
    quantization_seconds = 0.0
    eos_ids = eos_token_ids(model, tokenizer)

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        dense_breakdown = past_key_values_memory_breakdown(past_key_values)
        packed_estimate = (
            turboquant_mse_packed_bytes(
                num_vectors_per_kind=dense_breakdown["num_key_value_vectors_per_kind"],
                vector_dimension=dense_breakdown["vector_dimension"],
                bits=qmse_bits,
            )
            if variant in {"qmse", "qmse_packed"}
            else None
        )
        if variant == "qmse":
            quant_started = time.monotonic()
            past_key_values = quantize_past_key_values_mse(
                past_key_values,
                bits=qmse_bits,
                seed=rotation_seed,
            )
            quantization_seconds += time.monotonic() - quant_started
            packed_actual = None
            recon_quality = None
        elif variant == "qmse_packed":
            quant_started = time.monotonic()
            packed_cache = build_packed_mse_cache(
                past_key_values,
                bits=qmse_bits,
                seed=rotation_seed,
                num_outlier_channels=num_outlier_channels,
                outlier_extra_bits=outlier_extra_bits,
                use_qjl_keys=use_qjl_keys,
                quantize_decode=quantize_decode,
                norm_guard=norm_guard,
            )
            recon_quality = verify_packed_reconstruction(past_key_values, packed_cache)
            del past_key_values
            past_key_values = packed_cache
            packed_actual = packed_cache_storage_breakdown(packed_cache)
            quantization_seconds += time.monotonic() - quant_started
        else:
            packed_actual = None
            recon_quality = None

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        generated_tokens = [next_token]
        attention_mask = inputs.get("attention_mask")
        post_cache_setup = gpu_current_memory_bytes()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for _ in range(max_new_tokens - 1):
            if eos_ids and int(next_token[0, 0].item()) in eos_ids:
                break
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=-1,
                )
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            if variant == "qmse":
                quant_started = time.monotonic()
                past_key_values = quantize_past_key_values_mse(
                    past_key_values,
                    bits=qmse_bits,
                    seed=rotation_seed,
                    token_slice=slice(-1, None),
                )
                quantization_seconds += time.monotonic() - quant_started
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token)

    completion_tokens = torch.cat(generated_tokens, dim=-1)
    text = tokenizer.decode(completion_tokens[0], skip_special_tokens=True).strip()
    generation_seconds = time.monotonic() - decode_started
    metrics = GenerationMetrics(
        variant=variant,
        qmse_bits=qmse_bits if variant in {"qmse", "qmse_packed"} else None,
        quantization_seconds=round(quantization_seconds, 4),
        generation_seconds=round(generation_seconds, 4),
        prompt_tokens=int(inputs["input_ids"].shape[-1]),
        completion_tokens=int(completion_tokens.shape[-1]),
        prefill_cache=dense_breakdown,
        turboquant_mse_packed_estimate=packed_estimate,
        turboquant_mse_packed_actual=packed_actual,
        post_cache_setup_gpu_memory=post_cache_setup,
        gpu_peak_memory=gpu_peak_memory_bytes(),
        reconstruction_quality=recon_quality,
    )
    return GenerationOutput(text=text, metrics=metrics)

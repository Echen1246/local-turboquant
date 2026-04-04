from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from turboquant.runtime.generation import GenerationMetrics


@dataclass(frozen=True)
class TelemetrySummary:
    variant: str
    qmse_bits: int | None
    prompt_tokens: int
    completion_tokens: int
    dense_kv_bytes: int
    packed_estimate_bytes: int | None
    packed_actual_bytes: int | None
    payload_savings_fraction: float | None
    payload_savings_percent: float | None
    post_cache_setup_allocated_bytes: int | None
    post_cache_setup_reserved_bytes: int | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    generation_seconds: float
    quantization_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_generation_metrics(metrics: GenerationMetrics | dict[str, Any]) -> TelemetrySummary:
    metrics_dict = metrics.to_dict() if isinstance(metrics, GenerationMetrics) else metrics
    prefill_cache = metrics_dict["prefill_cache"]
    packed_estimate = metrics_dict.get("turboquant_mse_packed_estimate")
    packed_actual = metrics_dict.get("turboquant_mse_packed_actual")
    post_cache_setup = metrics_dict.get("post_cache_setup_gpu_memory")
    gpu_peak = metrics_dict.get("gpu_peak_memory")

    dense_kv_bytes = int(prefill_cache["dense_kv_bytes"])
    packed_estimate_bytes = (
        int(packed_estimate["packed_kv_bytes"])
        if packed_estimate is not None
        else None
    )
    packed_actual_bytes = (
        int(packed_actual["packed_total_bytes"])
        if packed_actual is not None
        else None
    )
    effective_packed_bytes = packed_actual_bytes if packed_actual_bytes is not None else packed_estimate_bytes
    payload_savings_fraction = None
    payload_savings_percent = None
    if effective_packed_bytes is not None and dense_kv_bytes > 0:
        payload_savings_fraction = 1.0 - (effective_packed_bytes / dense_kv_bytes)
        payload_savings_percent = payload_savings_fraction * 100.0

    return TelemetrySummary(
        variant=str(metrics_dict["variant"]),
        qmse_bits=metrics_dict.get("qmse_bits"),
        prompt_tokens=int(metrics_dict["prompt_tokens"]),
        completion_tokens=int(metrics_dict["completion_tokens"]),
        dense_kv_bytes=dense_kv_bytes,
        packed_estimate_bytes=packed_estimate_bytes,
        packed_actual_bytes=packed_actual_bytes,
        payload_savings_fraction=payload_savings_fraction,
        payload_savings_percent=payload_savings_percent,
        post_cache_setup_allocated_bytes=(
            int(post_cache_setup["allocated_bytes"])
            if post_cache_setup is not None and post_cache_setup.get("allocated_bytes") is not None
            else None
        ),
        post_cache_setup_reserved_bytes=(
            int(post_cache_setup["reserved_bytes"])
            if post_cache_setup is not None and post_cache_setup.get("reserved_bytes") is not None
            else None
        ),
        peak_allocated_bytes=(
            int(gpu_peak["peak_allocated_bytes"])
            if gpu_peak is not None and gpu_peak.get("peak_allocated_bytes") is not None
            else None
        ),
        peak_reserved_bytes=(
            int(gpu_peak["peak_reserved_bytes"])
            if gpu_peak is not None and gpu_peak.get("peak_reserved_bytes") is not None
            else None
        ),
        generation_seconds=float(metrics_dict["generation_seconds"]),
        quantization_seconds=float(metrics_dict["quantization_seconds"]),
    )

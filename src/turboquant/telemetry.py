from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from turboquant.runtime.generation import GenerationMetrics


def _fmt_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


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

    def format(self, compact: bool = False) -> str:
        """Return a human-readable multi-line summary string."""
        if compact:
            return self._format_compact()
        return self._format_full()

    def _format_compact(self) -> str:
        savings = f"{self.payload_savings_percent:.1f}%" if self.payload_savings_percent else "n/a"
        gen_s = f"{self.generation_seconds:.2f}s"
        quant_s = f"{self.quantization_seconds:.3f}s"
        tok_s = (
            f"{self.completion_tokens / self.generation_seconds:.1f} tok/s"
            if self.generation_seconds > 0
            else "n/a"
        )
        mode = f"{self.qmse_bits}-bit" if self.qmse_bits else self.variant
        return f"[TurboQuant] {mode} | {savings} savings | {gen_s} ({tok_s}) | quant {quant_s}"

    def _format_full(self) -> str:
        lines: list[str] = []
        lines.append("")
        lines.append("  TurboQuant Telemetry")
        lines.append("  " + "─" * 50)

        mode = f"{self.qmse_bits}-bit {self.variant}" if self.qmse_bits else self.variant
        savings = f"{self.payload_savings_percent:.1f}%" if self.payload_savings_percent else "n/a"
        tok_s = (
            f"{self.completion_tokens / self.generation_seconds:.1f}"
            if self.generation_seconds > 0
            else "n/a"
        )

        lines.append("")
        lines.append(f"    Mode:               {mode}")
        lines.append(f"    KV savings:         {savings}")
        lines.append(f"    Generation:         {self.generation_seconds:.2f}s  ({tok_s} tok/s)")
        lines.append(f"    Quantization:       {self.quantization_seconds:.3f}s")

        lines.append("")
        lines.append(f"    Prompt tokens:      {self.prompt_tokens}")
        lines.append(f"    Completion tokens:  {self.completion_tokens}")

        lines.append("")
        lines.append(f"    Dense KV cache:     {_fmt_bytes(self.dense_kv_bytes)}")
        packed = self.packed_actual_bytes or self.packed_estimate_bytes
        lines.append(f"    Packed KV cache:    {_fmt_bytes(packed)}")

        if self.post_cache_setup_allocated_bytes is not None:
            lines.append("")
            lines.append(f"    GPU after setup:    {_fmt_bytes(self.post_cache_setup_allocated_bytes)}")
        if self.peak_allocated_bytes is not None:
            lines.append(f"    GPU peak:           {_fmt_bytes(self.peak_allocated_bytes)}")

        lines.append("")
        return "\n".join(lines)


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

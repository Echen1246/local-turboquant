from __future__ import annotations

import argparse
import json
from typing import Any

from turboquant import TurboQuantSession


def _run_variant(
    *,
    model: str,
    prompt: str,
    variant: str,
    bits: int,
    max_new_tokens: int,
    attn_implementation: str,
    dtype: str,
    device_map: str,
    num_outlier_channels: int = 0,
    outlier_extra_bits: int = 1,
) -> dict[str, Any]:
    session = TurboQuantSession.from_pretrained(
        model,
        variant=variant,
        bits=bits,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        num_outlier_channels=num_outlier_channels,
        outlier_extra_bits=outlier_extra_bits,
    )
    text = session.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    return {
        "variant": variant,
        "text": text,
        "metrics": session.last_metrics(),
        "telemetry": session.last_telemetry(),
        "compatibility": session.compatibility_report(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Local TurboQuant smoke test on a small Transformers model.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--prompt",
        default="Explain KV cache compression in one short paragraph.",
    )
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--num-outlier-channels", type=int, default=0)
    parser.add_argument("--outlier-extra-bits", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Only run qmse_packed instead of baseline + qmse_packed.",
    )
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    if not args.skip_baseline:
        results.append(
            _run_variant(
                model=args.model,
                prompt=args.prompt,
                variant="baseline",
                bits=args.bits,
                max_new_tokens=args.max_new_tokens,
                attn_implementation=args.attn_implementation,
                dtype=args.dtype,
                device_map=args.device_map,
            )
        )
    results.append(
        _run_variant(
            model=args.model,
            prompt=args.prompt,
            variant="qmse_packed",
            bits=args.bits,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation,
            dtype=args.dtype,
            device_map=args.device_map,
            num_outlier_channels=args.num_outlier_channels,
            outlier_extra_bits=args.outlier_extra_bits,
        )
    )
    print(json.dumps({"model": args.model, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

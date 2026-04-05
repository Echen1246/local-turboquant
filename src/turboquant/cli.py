from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from turboquant import (
    TurboQuantSession,
    inspect_transformers_model_compatibility,
    load_transformers_model,
)
from turboquant.adapters.transformers import TransformersLoadConfig

_SUPPORTED_BITS = [2, 3, 4]

_VARIANTS = {
    "qmse_packed": "Packed Q_mse cache (default, recommended)",
    "qmse": "Dense reconstructed Q_mse cache",
    "baseline": "No quantization (for comparison)",
}

_TESTED_MODELS = [
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "status": "fully working",
        "notes": "80% savings at 73K ctx, all layers quantize cleanly",
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "status": "works with norm guard",
        "notes": "3/28 layers kept dense due to extreme key norms, 71.5% savings",
    },
]


def _format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _read_prompt(args) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text()
    raise ValueError("Either --prompt or --prompt-file is required.")


def _common_load_kwargs(args) -> dict[str, Any]:
    return {
        "revision": args.revision,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
        "token": args.token,
        "cache_dir": args.cache_dir,
    }


def _handle_info(args) -> int:
    from turboquant import __version__

    if args.json:
        _print_json({
            "version": __version__,
            "variants": _VARIANTS,
            "supported_bits": _SUPPORTED_BITS,
            "tested_models": _TESTED_MODELS,
        })
        return 0

    print(f"TurboQuant v{__version__}")
    print()
    print("Quantization modes:")
    for name, desc in _VARIANTS.items():
        print(f"  {name:<14} {desc}")
    print()
    print(f"Supported bit widths: {', '.join(str(b) for b in _SUPPORTED_BITS)}")
    print()
    print("Bit width guide:")
    print("  4-bit  near-lossless (cosine sim ~0.995), ~75% KV savings")
    print("  3-bit  very good (cosine sim ~0.983), ~80% KV savings")
    print("  2-bit  experimental, visibly lossy")
    print()
    print("Tested models:")
    for m in _TESTED_MODELS:
        print(f"  {m['id']}")
        print(f"    status: {m['status']}")
        print(f"    {m['notes']}")
    print()
    print("Key flags:")
    print("  --use-qjl-keys      Enable Q_prod (QJL sign sketch for keys)")
    print("  --quantize-decode    Quantize decode-phase tokens too")
    print("  --no-norm-guard      Disable per-layer norm guard (paper-faithful)")
    return 0


def _handle_telemetry(args) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1
    data = json.loads(path.read_text())

    telemetry = data.get("telemetry")
    metrics = data.get("metrics")
    if telemetry is None and metrics is None:
        print("No telemetry or metrics found in the JSON file.")
        print("Run with `turboquant run --json` to produce a file with telemetry.")
        return 1

    variant = data.get("variant", "unknown")
    bits = data.get("bits")
    model = data.get("model", "unknown")

    print(f"Model:   {model}")
    print(f"Variant: {variant}")
    if bits is not None:
        print(f"Bits:    {bits}")
    print()

    if telemetry is not None:
        dense = telemetry.get("dense_kv_bytes")
        packed = telemetry.get("packed_actual_bytes") or telemetry.get("packed_estimate_bytes")
        savings = telemetry.get("payload_savings_percent")

        print("Cache compression:")
        print(f"  Dense KV cache:  {_format_bytes(dense)}")
        print(f"  Packed KV cache: {_format_bytes(packed)}")
        if savings is not None:
            print(f"  Savings:         {savings:.1f}%")
        print()

        post_setup = telemetry.get("post_cache_setup_allocated_bytes")
        peak = telemetry.get("peak_allocated_bytes")
        print("GPU memory:")
        print(f"  After cache setup: {_format_bytes(post_setup)}")
        print(f"  Peak during gen:   {_format_bytes(peak)}")
        print()

        gen_s = telemetry.get("generation_seconds")
        quant_s = telemetry.get("quantization_seconds")
        prompt_tok = telemetry.get("prompt_tokens")
        comp_tok = telemetry.get("completion_tokens")
        print("Timing:")
        if prompt_tok is not None:
            print(f"  Prompt tokens:       {prompt_tok}")
        if comp_tok is not None:
            print(f"  Completion tokens:   {comp_tok}")
        if quant_s is not None:
            print(f"  Quantization time:   {quant_s:.3f}s")
        if gen_s is not None:
            print(f"  Generation time:     {gen_s:.3f}s")
            if comp_tok and gen_s > 0:
                print(f"  Tokens/sec:          {comp_tok / gen_s:.1f}")

    if metrics is not None:
        recon = metrics.get("reconstruction_quality")
        if recon is not None:
            print()
            print("Reconstruction quality:")
            k_cos = recon.get("avg_key_cosine_sim")
            v_cos = recon.get("avg_val_cosine_sim")
            k_mse = recon.get("avg_key_mse")
            v_mse = recon.get("avg_val_mse")
            dense_layers = recon.get("dense_layer_count")
            quant_layers = recon.get("quantized_layer_count")
            if k_cos is not None:
                print(f"  Key cosine sim:      {k_cos:.6f}")
            if v_cos is not None:
                print(f"  Value cosine sim:    {v_cos:.6f}")
            if k_mse is not None:
                print(f"  Key MSE:             {k_mse:.6f}")
            if v_mse is not None:
                print(f"  Value MSE:           {v_mse:.6f}")
            if dense_layers is not None:
                print(f"  Dense layers:        {dense_layers}")
            if quant_layers is not None:
                print(f"  Quantized layers:    {quant_layers}")

    return 0


def _handle_inspect(args) -> int:
    tokenizer, model = load_transformers_model(
        TransformersLoadConfig(
            model_id_or_path=args.model,
            **_common_load_kwargs(args),
        )
    )
    _ = tokenizer
    report = inspect_transformers_model_compatibility(model).to_dict()
    if args.json:
        _print_json(report)
        return 0

    print(f"model: {args.model}")
    print(f"backend: {report['backend']}")
    print(f"compatible: {report['compatible']}")
    if report["reasons"]:
        print("reasons:")
        for item in report["reasons"]:
            print(f"- {item}")
    if report["warnings"]:
        print("warnings:")
        for item in report["warnings"]:
            print(f"- {item}")
    if report["details"]:
        print("details:")
        for key, value in report["details"].items():
            print(f"- {key}: {value}")
    return 0


def _handle_run(args) -> int:
    prompt = _read_prompt(args)
    session = TurboQuantSession.from_pretrained(
        args.model,
        variant=args.variant,
        bits=args.bits,
        rotation_seed=args.rotation_seed,
        num_outlier_channels=args.num_outlier_channels,
        outlier_extra_bits=args.outlier_extra_bits,
        use_qjl_keys=args.use_qjl_keys,
        quantize_decode=args.quantize_decode,
        norm_guard=not args.no_norm_guard,
        **_common_load_kwargs(args),
    )
    text = session.generate(
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
    )
    metrics = session.last_metrics()
    telemetry = session.last_telemetry()
    payload = {
        "model": args.model,
        "variant": args.variant,
        "bits": args.bits if args.variant != "baseline" else None,
        "text": text,
        "metrics": metrics,
        "telemetry": telemetry,
    }
    if args.json:
        _print_json(payload)
        return 0

    print(text)
    if args.show_metrics and metrics is not None:
        print("\nmetrics:")
        _print_json(metrics)
    if args.show_telemetry and telemetry is not None:
        print("\ntelemetry:")
        print(f"- dense_kv_bytes: {_format_bytes(telemetry['dense_kv_bytes'])}")
        print(f"- packed_estimate_bytes: {_format_bytes(telemetry['packed_estimate_bytes'])}")
        print(f"- packed_actual_bytes: {_format_bytes(telemetry['packed_actual_bytes'])}")
        print(f"- payload_savings_percent: {telemetry['payload_savings_percent']}")
        print(
            f"- post_cache_setup_allocated_bytes: "
            f"{_format_bytes(telemetry['post_cache_setup_allocated_bytes'])}"
        )
        print(f"- peak_allocated_bytes: {_format_bytes(telemetry['peak_allocated_bytes'])}")
        print(f"- generation_seconds: {telemetry['generation_seconds']}")
        print(f"- quantization_seconds: {telemetry['quantization_seconds']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turboquant",
        description="TurboQuant-style KV-cache compression for Hugging Face Transformers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser(
        "info", help="Show supported quantization settings, bit widths, and tested models."
    )
    info_parser.add_argument("--json", action="store_true", help="Print as JSON.")
    info_parser.set_defaults(func=_handle_info)

    telemetry_parser = subparsers.add_parser(
        "telemetry", help="Display formatted telemetry from a saved JSON run output."
    )
    telemetry_parser.add_argument("file", help="Path to a JSON file from `turboquant run --json`.")
    telemetry_parser.set_defaults(func=_handle_telemetry)

    def add_load_args(target) -> None:
        target.add_argument("--model", required=True, help="Hugging Face model ID or local model path.")
        target.add_argument("--revision", default=None, help="Optional model revision.")
        target.add_argument("--dtype", default="auto", help="Transformers dtype argument. Default: auto.")
        target.add_argument("--device-map", default="auto", help="Transformers device_map argument.")
        target.add_argument(
            "--attn-implementation",
            default="sdpa",
            help="Attention backend to request from Transformers. Default: sdpa.",
        )
        target.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Pass trust_remote_code=True when loading the model/tokenizer.",
        )
        target.add_argument("--token", default=None, help="Optional Hugging Face token.")
        target.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")

    inspect_parser = subparsers.add_parser("inspect", help="Load a model and report TurboQuant compatibility.")
    add_load_args(inspect_parser)
    inspect_parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    inspect_parser.set_defaults(func=_handle_inspect)

    run_parser = subparsers.add_parser("run", help="Run a prompt through a Transformers model with TurboQuant.")
    add_load_args(run_parser)
    prompt_group = run_parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", default=None, help="Prompt text.")
    prompt_group.add_argument("--prompt-file", default=None, help="Read the prompt from a file.")
    run_parser.add_argument(
        "--variant",
        default="qmse_packed",
        choices=["baseline", "qmse", "qmse_packed"],
        help="Generation variant to use. Default: qmse_packed.",
    )
    run_parser.add_argument("--bits", type=int, default=3, help="Quantization bits for qmse variants.")
    run_parser.add_argument("--rotation-seed", type=int, default=0, help="Rotation seed. Default: 0.")
    run_parser.add_argument(
        "--num-outlier-channels",
        type=int,
        default=0,
        help="Number of head_dim channels to treat as outliers and quantize at higher precision.",
    )
    run_parser.add_argument(
        "--outlier-extra-bits",
        type=int,
        default=1,
        help="Additional bits to allocate to outlier channels beyond the base bit width.",
    )
    run_parser.add_argument(
        "--use-qjl-keys",
        action="store_true",
        help="Enable QJL residual correction for keys (TurboQuant_prod). Gives unbiased attention logits.",
    )
    run_parser.add_argument(
        "--quantize-decode",
        action="store_true",
        help="Also quantize tokens generated during decode (increases compression, may reduce quality).",
    )
    run_parser.add_argument(
        "--no-norm-guard",
        action="store_true",
        help="Disable per-layer norm guard (all layers get quantized, paper-faithful).",
    )
    run_parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generated tokens.")
    run_parser.add_argument("--json", action="store_true", help="Print full output as JSON.")
    run_parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Print the raw generation metrics after the response.",
    )
    run_parser.add_argument(
        "--show-telemetry",
        action="store_true",
        help="Print a concise cache/memory telemetry summary after the response.",
    )
    run_parser.set_defaults(func=_handle_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

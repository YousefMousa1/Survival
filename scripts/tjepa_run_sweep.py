#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def run_one(python_bin, pipeline, config_path, base_args, out_dir):
    run_name = config_path.stem
    run_out = Path(out_dir) / run_name
    run_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_bin,
        pipeline,
        "--output-dir",
        str(run_out),
        "--tjepa-config",
        str(config_path),
    ] + base_args

    result = subprocess.run(cmd, capture_output=True, text=True)
    metrics_path = run_out / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    return result.returncode, metrics, result.stderr


def main():
    parser = argparse.ArgumentParser(description="Run a T-JEPA config sweep.")
    parser.add_argument("--configs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--pipeline", default="survival_pipeline.py")
    parser.add_argument(
        "--base-args",
        nargs="*",
        default=[],
        help="Extra args to pass to survival_pipeline.py",
    )
    args = parser.parse_args()

    configs = sorted(Path(args.configs_dir).glob("*.json"))
    if not configs:
        raise SystemExit("No config files found.")

    summary = []
    for config_path in configs:
        code, metrics, stderr = run_one(
            args.python_bin, args.pipeline, config_path, args.base_args, args.output_dir
        )
        record = {
            "config": config_path.name,
            "returncode": code,
            "stderr": stderr.strip(),
        }
        record.update(metrics)
        summary.append(record)

    out_path = Path(args.output_dir) / "sweep_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote sweep summary to {out_path}")


if __name__ == "__main__":
    main()

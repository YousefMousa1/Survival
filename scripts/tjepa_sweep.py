#!/usr/bin/env python3
import argparse
import itertools
import json
from pathlib import Path


def parse_grid(grid_args):
    grid = {}
    for item in grid_args:
        if "=" not in item:
            raise ValueError(f"Invalid grid item: {item}")
        key, values = item.split("=", 1)
        parsed = []
        for raw in values.split(","):
            raw = raw.strip()
            if raw.lower() == "null":
                parsed.append(None)
            elif raw.lower() in ("true", "false"):
                parsed.append(raw.lower() == "true")
            else:
                try:
                    if "." in raw:
                        parsed.append(float(raw))
                    else:
                        parsed.append(int(raw))
                except ValueError:
                    parsed.append(raw)
        grid[key] = parsed
    return grid


def main():
    parser = argparse.ArgumentParser(description="Create T-JEPA sweep configs.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        help="Grid param in form key=v1,v2,v3. Can be repeated.",
    )
    args = parser.parse_args()

    base = json.loads(Path(args.base_config).read_text())
    grid = parse_grid(args.grid)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    configs = []
    for combo in itertools.product(*values):
        cfg = base.copy()
        for k, v in zip(keys, combo):
            cfg[k] = v
        configs.append(cfg)

    for idx, cfg in enumerate(configs, start=1):
        name_parts = [f"{k}-{cfg[k]}" for k in keys]
        name = "__".join(name_parts) if name_parts else f"config_{idx}"
        path = out_dir / f"{idx:03d}__{name}.json"
        path.write_text(json.dumps(cfg, indent=2))

    print(f"Wrote {len(configs)} configs to {out_dir}")


if __name__ == "__main__":
    main()

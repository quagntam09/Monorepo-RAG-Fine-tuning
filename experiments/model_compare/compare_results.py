from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_item(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Invalid --result value '{raw}'. Use format label=path.json")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Invalid label in '{raw}'")
    return label, Path(path.strip())


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare QA metrics across multiple fine-tuned models")
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Model result in label=path.json format (can be passed multiple times)",
    )
    parser.add_argument("--output-json", default=None, help="Optional output path for merged comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = []
    for item in args.result:
        label, path = _parse_item(item)
        metrics = _load_metrics(path)
        rows.append(
            {
                "model": label,
                "source": str(path),
                "exact_match": _to_float(metrics.get("exact_match")),
                "f1": _to_float(metrics.get("f1")),
                "loss": _to_float(metrics.get("loss")),
                "span_exact_match": _to_float(metrics.get("span_exact_match")),
            }
        )

    rows.sort(key=lambda row: row["f1"], reverse=True)

    print("| model | exact_match | f1 | loss | span_exact_match |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {row['exact_match']:.2f} | {row['f1']:.2f} | "
            f"{row['loss']:.4f} | {row['span_exact_match']:.4f} |"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

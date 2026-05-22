from __future__ import annotations

import argparse
import json
from pathlib import Path


def flatten_squad_records(raw_data: dict) -> list[dict]:
    rows: list[dict] = []
    for article in raw_data.get("data", []):
        title = article.get("title")
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])
                rows.append(
                    {
                        "id": qa.get("id"),
                        "title": title,
                        "question": qa.get("question", ""),
                        "context": context,
                        "answers": {
                            "text": [a.get("text", "") for a in answers],
                            "answer_start": [a.get("answer_start", -1) for a in answers],
                        },
                        "is_impossible": bool(qa.get("is_impossible", False)),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten nested SQuAD JSON to JSONL")
    parser.add_argument("--input_file", type=str, required=True, help="Path to nested SQuAD JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    rows = flatten_squad_records(raw_data=raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Flattened {len(rows)} QA rows -> {output_path}")


if __name__ == "__main__":
    main()

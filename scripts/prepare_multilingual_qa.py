#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


DATASET_ALIASES = {
    "squad": "rajpurkar/squad",
    "squad_v2": "rajpurkar/squad_v2",
}


def _normalize_dataset_id(dataset_id: str) -> str:
    normalized = (dataset_id or "").strip()
    if not normalized:
        raise ValueError("Dataset id must not be empty.")
    return DATASET_ALIASES.get(normalized, normalized)


def _to_dataset_dict(dataset_obj: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(dataset_obj, DatasetDict):
        return dataset_obj
    if isinstance(dataset_obj, Dataset):
        return DatasetDict({"train": dataset_obj})
    raise TypeError(f"Unsupported dataset type: {type(dataset_obj)!r}")


def _resolve_train_validation(
    dataset_obj: Dataset | DatasetDict,
    *,
    seed: int,
    validation_ratio: float,
) -> tuple[Dataset, Dataset]:
    dataset_dict = _to_dataset_dict(dataset_obj)
    if "train" not in dataset_dict:
        raise ValueError("Dataset must contain 'train' split.")
    train_split = dataset_dict["train"]

    for validation_name in ("validation", "dev", "val", "test"):
        if validation_name in dataset_dict:
            return train_split, dataset_dict[validation_name]

    split = train_split.train_test_split(test_size=validation_ratio, seed=seed)
    return split["train"], split["test"]


def _normalize_answers(raw_answers: Any) -> tuple[list[str], list[int]]:
    texts: list[Any] = []
    starts: list[Any] = []

    if isinstance(raw_answers, dict):
        texts = list(raw_answers.get("text", []) or [])
        starts = list(raw_answers.get("answer_start", []) or [])
    elif isinstance(raw_answers, list):
        for answer in raw_answers:
            if not isinstance(answer, dict):
                continue
            texts.append(answer.get("text", ""))
            starts.append(answer.get("answer_start", -1))
    else:
        return [], []

    normalized_texts: list[str] = []
    normalized_starts: list[int] = []
    for text, start in zip(texts, starts):
        answer_text = str(text or "").strip()
        if not answer_text:
            continue
        try:
            answer_start = int(start)
        except (TypeError, ValueError):
            continue
        if answer_start < 0:
            continue
        normalized_texts.append(answer_text)
        normalized_starts.append(answer_start)
    return normalized_texts, normalized_starts


def _normalize_row(row: dict[str, Any], *, lang: str, index: int) -> dict[str, Any] | None:
    question = str(row.get("question", "") or "").strip()
    context = str(row.get("context", "") or "").strip()
    if not question or not context:
        return None

    answer_texts, answer_starts = _normalize_answers(row.get("answers"))
    is_impossible = bool(row.get("is_impossible", False))
    if is_impossible:
        answer_texts = []
        answer_starts = []
    elif not answer_texts:
        # Keep rows without valid answer as impossible for a stable schema.
        is_impossible = True

    row_id = row.get("id")
    normalized_id = str(row_id).strip() if row_id is not None else ""
    if not normalized_id:
        normalized_id = f"{lang}-{index:08d}"

    return {
        "id": normalized_id,
        "question": question,
        "context": context,
        "answers": {
            "text": answer_texts,
            "answer_start": answer_starts,
        },
        "is_impossible": is_impossible,
        "language": lang,
    }


def _normalize_split(split: Dataset, *, lang: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, raw_row in enumerate(split):
        normalized = _normalize_row(raw_row, lang=lang, index=idx)
        if normalized is not None:
            rows.append(normalized)
    return rows


def _sample_rows(rows: list[dict[str, Any]], *, target_size: int, rng: random.Random, allow_oversample: bool) -> list[dict[str, Any]]:
    if target_size <= 0:
        return []
    if len(rows) == 0:
        return []
    if target_size <= len(rows):
        return rng.sample(rows, target_size)
    if not allow_oversample:
        return list(rows)
    return [rows[rng.randrange(len(rows))] for _ in range(target_size)]


def _build_mixed_rows(
    en_rows: list[dict[str, Any]],
    vi_rows: list[dict[str, Any]],
    *,
    total_size: int,
    vi_ratio: float,
    rng: random.Random,
    allow_oversample: bool,
) -> list[dict[str, Any]]:
    vi_target = int(round(total_size * vi_ratio))
    en_target = max(0, total_size - vi_target)

    selected_en = _sample_rows(
        en_rows,
        target_size=en_target,
        rng=rng,
        allow_oversample=allow_oversample,
    )
    selected_vi = _sample_rows(
        vi_rows,
        target_size=vi_target,
        rng=rng,
        allow_oversample=allow_oversample,
    )
    mixed = selected_en + selected_vi
    rng.shuffle(mixed)
    return mixed


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare multilingual QA JSONL files for stage-1 mixed and stage-2 VI fine-tuning."
    )
    parser.add_argument("--en-dataset", default="rajpurkar/squad")
    parser.add_argument("--en-dataset-config", default="")
    parser.add_argument("--vi-dataset", default="taidng/UIT-ViQuAD2.0")
    parser.add_argument("--vi-dataset-config", default="")
    parser.add_argument("--output-dir", default="data/qa_multilingual")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--stage1-vi-ratio", type=float, default=0.5)
    parser.add_argument("--stage1-train-size", type=int, default=0)
    parser.add_argument("--stage1-validation-size", type=int, default=4000)
    parser.add_argument("--stage2-train-size", type=int, default=0)
    parser.add_argument("--stage2-validation-size", type=int, default=0)
    parser.add_argument("--eval-en-size", type=int, default=0)
    parser.add_argument("--eval-vi-size", type=int, default=0)
    parser.add_argument("--allow-oversample", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (0.0 <= args.stage1_vi_ratio <= 1.0):
        raise ValueError("--stage1-vi-ratio must be in [0, 1].")
    if not (0.0 < args.validation_ratio < 1.0):
        raise ValueError("--validation-ratio must be in (0, 1).")

    rng = random.Random(args.seed)
    cache_dir = args.cache_dir or None

    en_dataset_id = _normalize_dataset_id(args.en_dataset)
    vi_dataset_id = _normalize_dataset_id(args.vi_dataset)

    en_dataset = load_dataset(
        path=en_dataset_id,
        name=(args.en_dataset_config or None),
        cache_dir=cache_dir,
    )
    vi_dataset = load_dataset(
        path=vi_dataset_id,
        name=(args.vi_dataset_config or None),
        cache_dir=cache_dir,
    )

    en_train, en_validation = _resolve_train_validation(
        en_dataset,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
    )
    vi_train, vi_validation = _resolve_train_validation(
        vi_dataset,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
    )

    en_train_rows = _normalize_split(en_train, lang="en")
    en_validation_rows = _normalize_split(en_validation, lang="en")
    vi_train_rows = _normalize_split(vi_train, lang="vi")
    vi_validation_rows = _normalize_split(vi_validation, lang="vi")

    if args.stage1_train_size > 0:
        stage1_train_size = args.stage1_train_size
    elif args.allow_oversample:
        stage1_train_size = len(en_train_rows) + len(vi_train_rows)
    else:
        if args.stage1_vi_ratio in (0.0, 1.0):
            stage1_train_size = len(vi_train_rows) if args.stage1_vi_ratio == 1.0 else len(en_train_rows)
        else:
            en_cap = len(en_train_rows) / (1.0 - args.stage1_vi_ratio)
            vi_cap = len(vi_train_rows) / args.stage1_vi_ratio
            stage1_train_size = int(math.floor(min(en_cap, vi_cap)))
    if stage1_train_size <= 0:
        raise ValueError("Unable to build stage1 train set with current parameters.")

    stage1_validation_size = args.stage1_validation_size
    if stage1_validation_size <= 0:
        stage1_validation_size = min(len(en_validation_rows), len(vi_validation_rows)) * 2

    stage1_train_rows = _build_mixed_rows(
        en_train_rows,
        vi_train_rows,
        total_size=stage1_train_size,
        vi_ratio=args.stage1_vi_ratio,
        rng=rng,
        allow_oversample=args.allow_oversample,
    )
    stage1_validation_rows = _build_mixed_rows(
        en_validation_rows,
        vi_validation_rows,
        total_size=stage1_validation_size,
        vi_ratio=args.stage1_vi_ratio,
        rng=rng,
        allow_oversample=args.allow_oversample,
    )

    stage2_train_rows = vi_train_rows
    if args.stage2_train_size > 0:
        stage2_train_rows = _sample_rows(
            vi_train_rows,
            target_size=args.stage2_train_size,
            rng=rng,
            allow_oversample=args.allow_oversample,
        )

    stage2_validation_rows = vi_validation_rows
    if args.stage2_validation_size > 0:
        stage2_validation_rows = _sample_rows(
            vi_validation_rows,
            target_size=args.stage2_validation_size,
            rng=rng,
            allow_oversample=args.allow_oversample,
        )

    eval_en_rows = en_validation_rows
    if args.eval_en_size > 0:
        eval_en_rows = _sample_rows(
            en_validation_rows,
            target_size=args.eval_en_size,
            rng=rng,
            allow_oversample=args.allow_oversample,
        )

    eval_vi_rows = vi_validation_rows
    if args.eval_vi_size > 0:
        eval_vi_rows = _sample_rows(
            vi_validation_rows,
            target_size=args.eval_vi_size,
            rng=rng,
            allow_oversample=args.allow_oversample,
        )

    output_dir = Path(args.output_dir)
    outputs = {
        "stage1_train.jsonl": stage1_train_rows,
        "stage1_validation.jsonl": stage1_validation_rows,
        "stage2_vi_train.jsonl": stage2_train_rows,
        "stage2_vi_validation.jsonl": stage2_validation_rows,
        "eval_en_validation.jsonl": eval_en_rows,
        "eval_vi_validation.jsonl": eval_vi_rows,
    }

    for file_name, rows in outputs.items():
        _write_jsonl(output_dir / file_name, rows)

    summary = {
        "output_dir": str(output_dir),
        "en_dataset": en_dataset_id,
        "vi_dataset": vi_dataset_id,
        "stage1_train": len(stage1_train_rows),
        "stage1_validation": len(stage1_validation_rows),
        "stage2_vi_train": len(stage2_train_rows),
        "stage2_vi_validation": len(stage2_validation_rows),
        "eval_en_validation": len(eval_en_rows),
        "eval_vi_validation": len(eval_vi_rows),
        "stage1_vi_ratio": args.stage1_vi_ratio,
        "allow_oversample": args.allow_oversample,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

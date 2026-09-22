"""Data and prompt helpers for four-option visual question answering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

ANSWER_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3}
ID_TO_ANSWER = {value: key for key, value in ANSWER_TO_ID.items()}
IMAGE_TOKEN = "<image>"


def build_mcq_prompt(question: str, options: list[str], variant: str = "visual_evidence") -> str:
    """Format one image question so the model only needs to predict A, B, C, or D."""
    if len(options) != 4:
        raise ValueError("Exactly four answer options are required.")

    if variant == "baseline":
        instruction = (
            "You are answering a multiple-choice question about the image.\n"
            "Choose the single best option. Reply with only one letter: A, B, C, or D."
        )
    elif variant == "visual_evidence":
        instruction = (
            "You are answering a multiple-choice question about the image.\n"
            "Use the image to choose the single best option. "
            "Reply with only one letter: A, B, C, or D."
        )
    elif variant == "concise":
        instruction = (
            "Answer the following image-based multiple-choice question.\n"
            "Output only A, B, C, or D."
        )
    else:
        raise ValueError(f"Unknown prompt variant: {variant}")

    return (
        f"{IMAGE_TOKEN}\n"
        f"{instruction}\n\n"
        f"Question: {question}\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}\n"
        "Answer:\n"
    )


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_vqa_records(path: str | Path, prompt_variant: str = "visual_evidence") -> list[dict]:
    """Load JSONL rows and add the formatted model prompt to each record."""
    records = read_jsonl(path)
    result = []
    for record in records:
        item = dict(record)
        item["prompt"] = build_mcq_prompt(
            item["question"], item["options"], variant=prompt_variant
        )
        result.append(item)
    return result

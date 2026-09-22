"""Compare answer letters in two VQA JSONL files."""

from __future__ import annotations

import argparse

from multimodal_vqa.data import read_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("gold")
parser.add_argument("predictions")
args = parser.parse_args()

gold = read_jsonl(args.gold)
predicted = read_jsonl(args.predictions)
if len(gold) != len(predicted):
    raise SystemExit("Files contain different numbers of questions.")

correct = sum(a["answer"] == b["answer"] for a, b in zip(gold, predicted))
print(f"Accuracy: {correct / len(gold):.2%} ({correct}/{len(gold)})")

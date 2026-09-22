"""Print the number of trainable parameters stored in a lightweight checkpoint."""

from __future__ import annotations

import argparse

import torch

from multimodal_vqa.training import count_snapshot_parameters

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
args = parser.parse_args()

snapshot = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
print(f"Tensors: {len(snapshot):,}")
print(f"Parameters: {count_snapshot_parameters(snapshot):,}")

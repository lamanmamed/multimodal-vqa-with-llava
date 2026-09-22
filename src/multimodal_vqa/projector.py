"""Project LongCLIP visual features into Qwen's hidden size."""

from __future__ import annotations

import torch
from torch import nn


class GatedResidualProjector(nn.Module):
    """Map each visual token from vision_hidden dimensions to text_hidden dimensions."""

    def __init__(
        self,
        vision_hidden: int = 768,
        text_hidden: int = 1024,
        *,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden, text_hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(text_hidden, text_hidden)
        self.gate = nn.Linear(vision_hidden, text_hidden)
        self.residual = nn.Linear(vision_hidden, text_hidden, bias=False)
        self.norm = nn.LayerNorm(text_hidden)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        transformed = self.linear_1(image_features)
        transformed = self.activation(transformed)
        transformed = self.dropout(transformed)
        transformed = self.linear_2(transformed)

        gate = torch.sigmoid(self.gate(image_features))
        residual = self.residual(image_features)
        return self.norm(transformed * gate + residual)

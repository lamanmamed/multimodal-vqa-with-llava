"""Training utilities used by the parameter-efficient VQA setup."""

from __future__ import annotations

import torch


def answer_only_labels(input_ids: torch.Tensor) -> torch.Tensor:
    """Mask the prompt so causal-LM loss is applied only to the final answer token."""
    if input_ids.ndim != 1 or input_ids.numel() == 0:
        raise ValueError("input_ids must be a non-empty 1D tensor.")
    labels = torch.full_like(input_ids, -100)
    labels[-1] = input_ids[-1]
    return labels


def trainable_snapshot(model) -> dict[str, torch.Tensor]:
    """Copy only parameters that are updated during fine-tuning."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def count_snapshot_parameters(snapshot: dict[str, torch.Tensor]) -> int:
    return sum(value.numel() for value in snapshot.values())


def split_trainable_parameters(model):
    """Separate projector parameters from other trainable parameters such as LoRA adapters."""
    projector = list(model.model.multi_modal_projector.parameters())
    projector_ids = {id(parameter) for parameter in projector}
    adapters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in projector_ids
    ]
    return projector, adapters

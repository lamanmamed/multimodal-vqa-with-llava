"""Constrained answer scoring for A/B/C/D visual question answering."""

from __future__ import annotations

import torch

ANSWER_LETTERS = ("A", "B", "C", "D")


def answer_token_ids(tokenizer, device=None) -> torch.Tensor:
    """Return one tokenizer id for each valid answer letter."""
    ids = []
    for letter in ANSWER_LETTERS:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Answer {letter} is not a single token: {token_ids}")
        ids.append(token_ids[0])
    return torch.tensor(ids, device=device)


def score_answer_logits(next_token_logits: torch.Tensor, valid_token_ids: torch.Tensor) -> torch.Tensor:
    """Choose among A/B/C/D from a vocabulary-sized next-token logit tensor."""
    if next_token_logits.ndim != 2:
        raise ValueError("next_token_logits must have shape (batch, vocabulary).")
    if valid_token_ids.numel() != 4:
        raise ValueError("Exactly four answer token ids are required.")
    return next_token_logits[:, valid_token_ids].argmax(dim=-1)


def last_non_padding_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Find the final real token for each sequence, including left-padded batches."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be 2D.")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    return (attention_mask * positions).max(dim=1).values


def extract_next_token_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Read logits from the last non-padding prompt token in each batch item."""
    indices = last_non_padding_index(attention_mask)
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, indices, :]

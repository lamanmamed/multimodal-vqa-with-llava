"""Match captions to images with normalized vision-language embeddings."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .data import read_jsonl, write_jsonl


def cosine_match(text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
    """Return the best image index for each caption using cosine similarity."""
    if text_embeddings.ndim != 2 or image_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D tensors.")
    if text_embeddings.shape[1] != image_embeddings.shape[1]:
        raise ValueError("Text and image embeddings must have the same feature size.")

    text_embeddings = F.normalize(text_embeddings, dim=-1)
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    return (text_embeddings @ image_embeddings.T).argmax(dim=1)


def evaluate_alignment(gold_records: list[dict], predicted_records: list[dict]) -> float:
    """Compare (caption_id, image) pairs regardless of JSONL row order."""
    gold = {(row["caption_id"], row["image"]) for row in gold_records}
    predicted = {(row["caption_id"], row["image"]) for row in predicted_records}
    if not gold:
        raise ValueError("Gold alignment is empty.")
    return len(gold & predicted) / len(gold)


def attach_image_paths(
    vqa_records: list[dict],
    caption_to_image: dict[str, str],
) -> list[dict]:
    """Add the recovered image path to each question using its caption id."""
    output = []
    for record in vqa_records:
        caption_id = record["caption_id"]
        if caption_id not in caption_to_image:
            raise KeyError(f"No image match for caption id {caption_id}")
        row = dict(record)
        row["image"] = caption_to_image[caption_id]
        output.append(row)
    return output


def recover_image_paths_with_longclip(
    image_dir: str | Path,
    caption_files: list[str | Path],
    *,
    model_id: str = "AlpachinoNLP/LongCLIP-ViT-B-32",
    image_batch_size: int = 64,
    text_batch_size: int = 128,
    device: str | None = None,
) -> dict[str, str]:
    """Encode all images and captions with LongCLIP and match each caption to its nearest image."""
    try:
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError("Install the project with the 'models' extra.") from exc

    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        raise ValueError(f"No JPG images found in {image_dir}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    text_max_length = model.config.text_config.max_position_embeddings

    image_chunks = []
    with torch.no_grad():
        for start in range(0, len(image_files), image_batch_size):
            paths = image_files[start:start + image_batch_size]
            images = [Image.open(path).convert("RGB") for path in paths]
            batch = processor(images=images, return_tensors="pt").to(device)
            features = model.get_image_features(**batch).pooler_output
            image_chunks.append(F.normalize(features, dim=-1).cpu())
    image_embeddings = torch.cat(image_chunks)

    caption_ids: list[str] = []
    caption_texts: list[str] = []
    for path in caption_files:
        for record in read_jsonl(path):
            caption_ids.append(record["caption_id"])
            caption_texts.append(record["caption"])

    text_chunks = []
    with torch.no_grad():
        for start in range(0, len(caption_texts), text_batch_size):
            texts = caption_texts[start:start + text_batch_size]
            batch = processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=text_max_length,
            ).to(device)
            features = model.get_text_features(**batch).pooler_output
            text_chunks.append(F.normalize(features, dim=-1).cpu())
    text_embeddings = torch.cat(text_chunks)

    indices = cosine_match(text_embeddings, image_embeddings).tolist()
    return {
        caption_id: str(image_files[index].as_posix())
        for caption_id, index in zip(caption_ids, indices)
    }

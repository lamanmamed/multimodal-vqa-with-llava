"""Create full-image and cropped views for test-time averaging experiments."""

from __future__ import annotations

from PIL import Image


def make_multicrop_views(image: Image.Image, crop_ratio: float = 0.75) -> list[Image.Image]:
    if not 0 < crop_ratio <= 1:
        raise ValueError("crop_ratio must be in (0, 1].")

    image = image.convert("RGB")
    width, height = image.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2

    return [
        image,
        image.crop((left, top, left + crop_width, top + crop_height)),
        image.crop((0, 0, crop_width, crop_height)),
        image.crop((width - crop_width, 0, width, crop_height)),
        image.crop((0, height - crop_height, crop_width, height)),
        image.crop((width - crop_width, height - crop_height, width, height)),
    ]

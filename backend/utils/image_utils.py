from __future__ import annotations

import base64
import io
import os
from typing import Any

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, resize


def load_image_tensor(
    image_ref: str | None,
    *,
    image_root: str | None = None,
    image_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    width, height = int(image_size[0]), int(image_size[1])
    if not image_ref:
        return torch.zeros(3, height, width, dtype=torch.float32)

    image = _open_image(image_ref, image_root=image_root)
    image = image.convert("RGB")
    image = resize(image, [height, width])
    tensor = pil_to_tensor(image).float() / 255.0
    return tensor


def normalize_infer_images(images: Any) -> list[str]:
    if images is None:
        return []
    if isinstance(images, str):
        return [images]
    if isinstance(images, list):
        if images and all(isinstance(item, str) for item in images):
            return [str(item) for item in images]
        flattened: list[str] = []
        for item in images:
            if isinstance(item, str):
                flattened.append(item)
            elif isinstance(item, list):
                flattened.extend(str(child) for child in item if isinstance(child, str))
        return flattened
    return []


def build_infer_image_tensors(
    images: Any,
    *,
    num_cameras: int,
    image_size: tuple[int, int],
) -> list[torch.Tensor]:
    refs = normalize_infer_images(images)
    if num_cameras <= 0:
        return []
    refs = refs[:num_cameras] + [""] * max(0, num_cameras - len(refs))
    return [
        load_image_tensor(ref, image_size=image_size).unsqueeze(0)
        for ref in refs
    ]


def _open_image(image_ref: str, *, image_root: str | None = None) -> Image.Image:
    if image_ref.startswith("data:image") and "base64," in image_ref:
        _, raw = image_ref.split("base64,", 1)
        return Image.open(io.BytesIO(base64.b64decode(raw)))
    path = image_ref
    if image_root and not os.path.isabs(path):
        path = os.path.join(image_root, path)
    return Image.open(path)

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from PIL import Image, UnidentifiedImageError

from config.settings import (
    ALLOWED_IMAGE_TYPES,
    ALLOWED_EXTENSIONS,
    MAX_IMAGE_SIZE_BYTES,
    MAX_IMAGE_DIMENSIONS,
)

logger = logging.getLogger(__name__)


def validate_uploaded_image(uploaded_file) -> Tuple[bool, str]:
    if uploaded_file is None:
        return False, "No file was uploaded."

    if uploaded_file.size > MAX_IMAGE_SIZE_BYTES:
        size_mb = uploaded_file.size / (1024 * 1024)
        limit_mb = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
        logger.warning(f"Rejected oversized upload: {size_mb:.1f}MB (limit {limit_mb}MB)")
        return False, f"File is too large ({size_mb:.1f} MB). Maximum allowed size is {limit_mb:.0f} MB."

    if uploaded_file.type not in ALLOWED_IMAGE_TYPES:
        logger.warning(f"Rejected unsupported MIME type: {uploaded_file.type}")
        return False, (
            f"Unsupported file type: '{uploaded_file.type}'. "
            "Please upload a JPG, PNG, or WebP image."
        )

    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Rejected unsupported extension: {ext}")
        return False, (
            f"Unsupported file extension: '{ext}'. "
            "Accepted formats: .jpg, .jpeg, .png, .webp"
        )

    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)

        img = Image.open(uploaded_file)
        w, h = img.size
        max_w, max_h = MAX_IMAGE_DIMENSIONS
        if w > max_w or h > max_h:
            logger.warning(f"Rejected oversized dimensions: {w}×{h}")
            return False, (
                f"Image dimensions ({w}×{h} px) exceed the maximum "
                f"({max_w}×{max_h} px). Please resize your image."
            )

        uploaded_file.seek(0)

    except UnidentifiedImageError:
        logger.warning(f"PIL could not identify file as an image: {uploaded_file.name}")
        return False, "This file does not appear to be a valid image. Please upload a different file."
    except Exception as exc:
        logger.warning(f"Image validation failed unexpectedly: {exc}")
        return False, "The file could not be read. It may be corrupted."

    logger.debug(f"Image validated successfully: {uploaded_file.name} ({w}×{h} px)")
    return True, ""

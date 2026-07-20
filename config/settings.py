from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(PROJECT_ROOT / "models" / "best_model.pth"),
)
MODEL_METADATA_PATH = os.getenv(
    "MODEL_METADATA_PATH",
    str(PROJECT_ROOT / "models" / "model_metadata.json"),
)
CLASS_LABELS_PATH = os.getenv(
    "CLASS_LABELS_PATH",
    str(PROJECT_ROOT / "models" / "class_labels.pkl"),
)

ANIMAL_INFO_PATH = os.getenv(
    "ANIMAL_INFO_PATH",
    str(PROJECT_ROOT / "data" / "animal_info.csv"),
)

IMG_SIZE: tuple[int, int] = (224, 224)

CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))

MARGIN_THRESHOLD: float = float(os.getenv("MARGIN_THRESHOLD", "0.20"))

TOP_K_PREDICTIONS: int = 5

MODEL_ARCHITECTURE: str = "ConvNeXt-Tiny"

MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024

MAX_IMAGE_DIMENSIONS: tuple[int, int] = (8000, 8000)

ALLOWED_IMAGE_TYPES: set[str] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}

ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CONSERVATION_STATUS_COLOURS: dict[str, str] = {
    "Extinct":                "#1a1a1a",
    "Extinct in the Wild":    "#3d0000",
    "Critically Endangered":  "#c0392b",
    "Endangered":             "#e67e22",
    "Vulnerable":             "#f1c40f",
    "Near Threatened":        "#2ecc71",
    "Least Concern":          "#27ae60",
    "Data Deficient":         "#7f8c8d",
    "Not Evaluated":          "#95a5a6",
}

CONSERVATION_STATUS_ICONS: dict[str, str] = {
    "Extinct":                "EX",
    "Extinct in the Wild":    "EW",
    "Critically Endangered":  "CR",
    "Endangered":             "EN",
    "Vulnerable":             "VU",
    "Near Threatened":        "NT",
    "Least Concern":          "LC",
    "Data Deficient":         "DD",
    "Not Evaluated":          "NE",
}

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

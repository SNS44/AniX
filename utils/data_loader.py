from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from config.settings import (
    ANIMAL_INFO_PATH,
    MODEL_METADATA_PATH,
    CONSERVATION_STATUS_COLOURS,
    CONSERVATION_STATUS_ICONS,
)

logger = logging.getLogger(__name__)


@st.cache_data
def load_animal_info(mtime: float = 0.0) -> pd.DataFrame:
    path = Path(ANIMAL_INFO_PATH)
    if not path.exists():
        logger.warning(f"animal_info.csv not found at {path} — returning empty DataFrame")
        return pd.DataFrame(columns=[
            "common_name", "scientific_name", "family", "habitat",
            "diet", "lifespan_wild", "weight_kg", "conservation_status",
            "geographic_range", "interesting_fact", "threats", "description",
        ])
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded animal_info.csv — {len(df)} species records")
        return df
    except Exception as exc:
        logger.error(f"Failed to load animal_info.csv: {exc}")
        return pd.DataFrame()


@st.cache_data
def load_model_metadata() -> dict:
    path = Path(MODEL_METADATA_PATH)
    if not path.exists():
        return {
            "version": "Not trained",
            "architecture": "EfficientNetV2B0",
            "validation_accuracy": None,
            "num_classes": 0,
            "trained_date": "N/A",
            "notes": "Train the model using notebooks/03_training.ipynb",
        }
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.error(f"Failed to load model_metadata.json: {exc}")
        return {}


def get_species_profile(animal_info_df: pd.DataFrame, label: str) -> Optional[pd.Series]:
    if animal_info_df.empty:
        return None

    mask = animal_info_df["common_name"].str.lower() == label.lower()
    matches = animal_info_df[mask]

    if matches.empty:
        logger.debug(f"No species profile found for: '{label}'")
        return None

    return matches.iloc[0]


def get_conservation_colour(status: str) -> str:
    return CONSERVATION_STATUS_COLOURS.get(status, "#6b7280")


def get_conservation_icon(status: str) -> str:
    return CONSERVATION_STATUS_ICONS.get(status, "?")

from __future__ import annotations

import io
import logging
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict import AnimalPredictor
from config.settings import ANIMAL_INFO_PATH, CONFIDENCE_THRESHOLD
from utils.data_loader import (
    get_conservation_colour,
    get_conservation_icon,
    get_species_profile,
    load_animal_info,
)
from utils.validators import validate_uploaded_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _lucide(paths: str, size: int = 18) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{paths}</svg>'
    )


ICON = {
    "brand":    _lucide('<circle cx="11" cy="4" r="2"/><circle cx="18" cy="8" r="2"/><circle cx="20" cy="16" r="2"/><circle cx="4" cy="8" r="2"/><circle cx="2" cy="16" r="2"/><path d="M7 18a4 4 0 0 0 4-4 4 4 0 0 0 4 4 4 4 0 0 1-4 4 4 4 0 0 1-4-4Z"/>', 32),
    "search":   _lucide('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>'),
    "globe":    _lucide('<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>'),
    "utensils": _lucide('<path d="M3 2v7c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2V2"/><path d="M7 2v20"/><path d="M21 15V2a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3Zm0 0v7"/>'),
    "scale":    _lucide('<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>'),
    "calendar": _lucide('<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/>'),
    "map_pin":  _lucide('<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>'),
    "alert":    _lucide('<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>'),
    "bulb":     _lucide('<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>'),
    "shield":   _lucide('<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'),
    "file":     _lucide('<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>'),
    "book":     _lucide('<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>'),
    "scan":     _lucide('<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3"/><path d="m16 16-1.5-1.5"/>'),
    "help":     _lucide('<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>'),
}
ICON_SEARCH_LG = _lucide('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>', 48)


def load_styles() -> None:
    style_path = PROJECT_ROOT / "style.css"
    if style_path.exists():
        with open(style_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>body { font-family: sans-serif; }</style>", unsafe_allow_html=True)


def glass_card(html: str, extra_class: str = "") -> None:
    st.markdown(
        f"<div class='glass-card {extra_class}'>{html}</div>",
        unsafe_allow_html=True,
    )


def section_header(eyebrow: str, title: str = "", desc: str = "") -> None:
    html = f"<div class='section-eyebrow'>{eyebrow}</div>"
    if title:
        html += f"<div class='section-title'>{title}</div>"
    if desc:
        html += f"<div class='section-desc'>{desc}</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_info_row(icon: str, label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="info-row">
            <div class="info-icon">{icon}</div>
            <div>
                <div class="info-label">{label}</div>
                <div class="info-value">{value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Species Profile Details", width="large")
def show_species_modal(species: pd.Series | dict) -> None:
    cons_status = str(species.get("conservation_status", ""))
    cons_colour = get_conservation_colour(cons_status)
    cons_icon   = get_conservation_icon(cons_status)

    st.markdown(
        f"""
        <div style="text-align: center; padding-bottom: 1rem; border-bottom: 1px solid var(--glass-border); margin-bottom: 1.5rem;">
            <h2 style="font-family: var(--font-display); margin: 0; color: var(--text); font-size: 2.2rem;">{species.get('common_name', '—')}</h2>
            <div style="font-style: italic; color: var(--accent); margin: 0.25rem 0 0.75rem 0; font-size: 1.1rem;">{species.get('scientific_name', '—')}</div>
            <div class="cons-badge" style="background:{cons_colour}22; color:{cons_colour}; border:1px solid {cons_colour}55; font-size: 0.85rem; padding: 0.35rem 1rem;">
                {cons_icon} {cons_status}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("<div class='section-eyebrow'>BIOLOGICAL PROFILE</div>", unsafe_allow_html=True)
        render_info_row(ICON["globe"], "Habitat", str(species.get("habitat", "—")))
        render_info_row(ICON["utensils"], "Diet", str(species.get("diet", "—")))
        
        weight_val = str(species.get("weight_kg", "—"))
        if weight_val != "nan" and weight_val != "—" and "kg" not in weight_val.lower():
            weight_val += " kg"
        render_info_row(ICON["scale"], "Weight", weight_val)
        
        render_info_row(ICON["calendar"], "Lifespan (Wild)", str(species.get("lifespan_wild", "—")))
        render_info_row(ICON["map_pin"], "Range", str(species.get("geographic_range", "—")))
        render_info_row(ICON["alert"], "Threats", str(species.get("threats", "—")))

    with col2:
        st.markdown("<div class='section-eyebrow'>DESCRIPTION</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:var(--text2); font-size:0.95rem; line-height:1.75; margin:0;'>{species.get('description', 'No description available.')}</p>", unsafe_allow_html=True)

        fact = species.get("interesting_fact")
        if fact and str(fact) != "nan" and str(fact) != "—":
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="glass-card" style="border-color: rgba(243, 156, 18, 0.3); background: rgba(243, 156, 18, 0.03); padding: 1rem;">
                    <div style="font-size:0.75rem; color:var(--gold); text-transform:uppercase; letter-spacing:2px; font-weight:700; margin-bottom:0.5rem; display:flex; align-items:center; gap:0.4rem;">{ICON['bulb']} Did You Know?</div>
                    <p style='color:var(--text2); font-size:0.9rem; line-height:1.65; margin:0;'>{fact}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    tax_fields = ["kingdom", "phylum", "class", "order", "family", "genus"]
    chips = " ".join(
        f"<span class='chip'>{f.capitalize()}: <b>{species.get(f, '—')}</b></span>"
        for f in tax_fields
        if species.get(f) and str(species.get(f)) != "nan" and str(species.get(f)) != "—"
    )
    if chips:
        st.markdown("<br><div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-eyebrow'>TAXONOMY</div>", unsafe_allow_html=True)
        st.markdown(chips, unsafe_allow_html=True)


if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_resource(show_spinner=False)
def get_predictor() -> AnimalPredictor | None:
    try:
        return AnimalPredictor(PROJECT_ROOT)
    except Exception as exc:
        logger.error(f"Failed to load predictor: {exc}")
        return None

@st.cache_data(show_spinner=False)
def get_animal_info() -> pd.DataFrame:
    try:
        mtime = Path(ANIMAL_INFO_PATH).stat().st_mtime
    except Exception:
        mtime = 0.0
    return load_animal_info(mtime)


load_styles()

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand">
            <div class="brand-logo">{ICON['brand']}</div>
            <div class="brand-text">
                <div class="brand-name">AniX</div>
                <div class="brand-sub">Wildlife Identification</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predictor = get_predictor()
    model_ready = predictor is not None
    status_label = "Status: Online" if model_ready else "Status: Offline"
    status_color_class = "status-online" if model_ready else "status-offline"
    st.markdown(
        f"""
        <div class="status-pill {status_color_class}">
            <span class="status-dot"></span>
            <span>{status_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Identify", "Encyclopedia"],
        key="nav_page",
        label_visibility="collapsed",
    )


if page == "Identify":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Biodiversity Intelligence</div>
            <div class="hero-title">Wildlife <span>Identification</span> Portal</div>
            <div class="hero-subtitle">
                Analyze and identify wildlife species.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_ready:
        st.error("Model best_model.pth not loaded. Verify it is located in the models/ directory.")
        st.stop()

    animal_info_df = get_animal_info()

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        section_header("INPUT ZONE", "Species Upload", "Submit an image file to begin inference.")
        
        uploaded_file = st.file_uploader(
            "Upload image file",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
            key="main_uploader",
        )

        st.markdown(
            """
            <div style="margin-top: 1rem; font-size: 0.8rem; color: var(--text3); line-height: 1.6;">
                Supported formats: JPEG, PNG, WebP, BMP<br>
                File size limit: 10 MB
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        identify_btn = st.button("Identify Species", use_container_width=True, disabled=(uploaded_file is None))

    with col_right:
        section_header("PREVIEW ZONE", "Image Preview", "Visual assessment frame of the uploaded sample.")

        if uploaded_file:
            is_valid, error_msg = validate_uploaded_image(uploaded_file)
            
            if not is_valid:
                st.error(f"Image validation error: {error_msg}")
            else:
                image = Image.open(uploaded_file).convert("RGB")
                
                if identify_btn:
                    with st.spinner("Executing neural inference..."):
                        temp_path = PROJECT_ROOT / "temp_prediction.jpg"
                        try:
                            image.save(temp_path, format="JPEG", quality=95)
                            result = predictor.predict(temp_path)
                        finally:
                            if temp_path.exists():
                                temp_path.unlink(missing_ok=True)

                    pred_label = result["prediction"]
                    confidence = result["confidence"]
                    
                    if pred_label == "Unclassified":
                        st.image(image, use_container_width=True)
                        st.markdown(
                            f"""
                            <div class="glass-card unclassified" style="margin-top: 1.5rem;">
                                <div class="unclassified-icon">{ICON_SEARCH_LG}</div>
                                <div class="unclassified-title">Species Unclassified</div>
                                <p style="color:var(--text2); margin:0;">Inference confidence is below safety thresholds.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        profile = get_species_profile(animal_info_df, pred_label)
                        scientific = profile["scientific_name"] if profile is not None else "N/A"
                        habitat = profile["habitat"] if profile is not None else "N/A"
                        diet = profile["diet"] if profile is not None else "N/A"
                        cons_status = profile["conservation_status"] if profile is not None else "N/A"
                        desc = profile["description"] if profile is not None else "No description available."
                        fact = profile["interesting_fact"] if profile is not None else ""

                        cons_colour = get_conservation_colour(cons_status)
                        cons_icon = get_conservation_icon(cons_status)

                        st.image(image, use_container_width=True)

                        fact_html = ""
                        if fact and str(fact) != "nan" and str(fact) != "—":
                            fact_html = (
                                f'<div class="info-row">'
                                f'<div class="info-icon">{ICON["bulb"]}</div>'
                                f'<div>'
                                f'<div class="info-label">Interesting Facts</div>'
                                f'<div class="info-value">{fact}</div>'
                                f'</div></div>'
                            )

                        details_html = (
                             f'<div class="glass-card scale-in" style="margin-top: 1.5rem;">'
                             f'<div class="pred-hero">'
                             f'<div class="pred-conf-label">CLASSIFICATION RESULT</div>'
                             f'<div class="pred-species">{pred_label}</div>'
                             f'<div class="pred-scientific">{scientific}</div>'
                             f'<div class="pred-confidence">{confidence:.1f}%</div>'
                             f'<div class="pred-conf-label">Confidence Score</div>'
                             f'</div>'
                             f'<div class="divider"></div>'
                             f'<div class="section-eyebrow">TAXONOMICAL & ECOLOGICAL PROFILE</div>'
                             f'<div class="info-row">'
                             f'<div class="info-icon">{ICON["shield"]}</div>'
                             f'<div>'
                             f'<div class="info-label">Conservation Status</div>'
                             f'<div class="cons-badge" style="background:{cons_colour}22; color:{cons_colour}; border:1px solid {cons_colour}44; margin-top:0.25rem;">'
                             f'{cons_icon} {cons_status}'
                             f'</div>'
                             f'</div>'
                             f'</div>'
                             f'<div class="info-row">'
                             f'<div class="info-icon">{ICON["globe"]}</div>'
                             f'<div>'
                             f'<div class="info-label">Habitat</div>'
                             f'<div class="info-value">{habitat}</div>'
                             f'</div>'
                             f'</div>'
                             f'<div class="info-row">'
                             f'<div class="info-icon">{ICON["utensils"]}</div>'
                             f'<div>'
                             f'<div class="info-label">Diet</div>'
                             f'<div class="info-value">{diet}</div>'
                             f'</div>'
                             f'</div>'
                             f'<div class="info-row">'
                             f'<div class="info-icon">{ICON["file"]}</div>'
                             f'<div>'
                             f'<div class="info-label">Description</div>'
                             f'<div class="info-value">{desc}</div>'
                             f'</div>'
                             f'</div>'
                             f'{fact_html}'
                             f'</div>'
                        )
                        st.markdown(details_html, unsafe_allow_html=True)
                else:
                    st.image(image, use_container_width=True)
        else:
            st.markdown(
                """
                <div class="glass-card" style="display:flex; justify-content:center; align-items:center; min-height:380px;">
                    <div style="color:var(--text3); font-size:1rem; font-weight:600; text-transform:uppercase; letter-spacing:1px;">Image Preview</div>
                </div>
                """,
                unsafe_allow_html=True
            )


elif page == "Encyclopedia":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Reference Library</div>
            <div class="hero-title">Species <span>Encyclopedia</span></div>
            <div class="hero-subtitle">
                Explore biological profiles, conservation status, and fascinating facts
                for every species in the AniX training dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    animal_info_df = get_animal_info()

    if animal_info_df.empty:
        st.markdown(
            "<div class='error-card'>Species database not found. Make sure <code>data/animal_info.csv</code> exists.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    filter_col1, filter_col2 = st.columns([2, 1], gap="medium")
    with filter_col1:
        search = st.text_input(
            "Search species",
            placeholder="Search by name (e.g. lion, tiger, eagle...)",
            label_visibility="collapsed",
        )
    with filter_col2:
        statuses = ["All Statuses"] + sorted(animal_info_df["conservation_status"].dropna().unique().tolist())
        status_filter = st.selectbox("Conservation status", statuses, label_visibility="collapsed")

    df_view = animal_info_df.copy()
    if search:
        mask = (
            df_view["common_name"].str.contains(search, case=False, na=False) |
            df_view["scientific_name"].str.contains(search, case=False, na=False)
        )
        df_view = df_view[mask]
    if status_filter != "All Statuses":
        df_view = df_view[df_view["conservation_status"] == status_filter]

    st.markdown(
        f"<div style='font-size:0.78rem; color:var(--text3); margin-bottom:1rem;'>Showing <b style='color:var(--accent);'>{len(df_view)}</b> of {len(animal_info_df)} species</div>",
        unsafe_allow_html=True,
    )

    if df_view.empty:
        glass_card(
            "<div style='text-align:center; padding:2rem;'>"
            f"<div style='font-size:3rem;'>{ICON_SEARCH_LG}</div>"
            "<div style='color:var(--text2); margin-top:0.75rem;'>No species match your search.</div>"
            "</div>"
        )
    else:
        COLS = 3
        for i in range(0, len(df_view), COLS):
            chunk = df_view.iloc[i:i+COLS]
            cols = st.columns(COLS, gap="medium")
            for col, (_, species_row) in zip(cols, chunk.iterrows()):
                name = species_row.get("common_name", "—")
                sci = species_row.get("scientific_name", "—")
                status = species_row.get("conservation_status", "—")
                desc = str(species_row.get("description", "No description available."))
                habitat_raw = str(species_row.get("habitat", "—") or "")
                habitat = habitat_raw[:72] + ("..." if len(habitat_raw) > 72 else "")
                
                cons_colour = get_conservation_colour(status)
                cons_icon = get_conservation_icon(status)
                
                with col:
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div class="species-common">{name}</div>
                            <div class="species-sci">{sci}</div>
                            <div style="margin-bottom:0.65rem;">
                                <span class="cons-badge" style="background:{cons_colour}22; color:{cons_colour}; border:1px solid {cons_colour}44; font-size:0.7rem; padding:0.22rem 0.65rem; margin-top:0;">
                                    {cons_icon} {status}
                                </span>
                            </div>
                            <div class="species-desc">{desc}</div>
                            <div class="species-habitat">Habitat: {habitat}</div>
                            """,
                            unsafe_allow_html=True
                        )
                        if st.button("View Details", key=f"btn_{name}", use_container_width=True):
                            show_species_modal(species_row)

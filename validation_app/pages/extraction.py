"""
Página 1 — Validación de extracción.

Permite comparar la calidad de extracción de PyMuPDF y Docling
contra el abstract oficial de la API de arXiv.
"""

import json
import random
from pathlib import Path

import streamlit as st

# Rutas relativas a la raíz del proyecto (donde se lanza el comando)
ROOT = Path(__file__).parent.parent.parent
EXTRACTED_PATH  = ROOT / "data/interim/extracted.json"
CATEGORIES_PATH = ROOT / "configs/categories.json"


# ── Carga de datos ────────────────────────────────────────────────────────────

@st.cache_data
def load_extracted() -> list[dict]:
    if not EXTRACTED_PATH.exists():
        return []
    with open(EXTRACTED_PATH) as f:
        return json.load(f)


@st.cache_data
def load_categories() -> dict[str, str]:
    """Devuelve {code: name}."""
    if not CATEGORIES_PATH.exists():
        return {}
    with open(CATEGORIES_PATH) as f:
        cats = json.load(f)
    return {c["code"]: c["name"] for c in cats}


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_color(score: float) -> str:
    if score >= 0.90:
        return "green"
    if score >= 0.70:
        return "orange"
    return "red"


def extraction_failed(article: dict, source: str) -> bool:
    status = article.get("extraction_status", {})
    if isinstance(status, dict):
        return status.get(source, "ok") == "failed"
    return False


# ── Página ────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🔍 Validación de extracción")
    st.caption("Compara el abstract extraído de PDF con el abstract oficial de la API de arXiv.")

    # Cargar datos
    extracted = load_extracted()
    categories = load_categories()

    if not extracted:
        st.warning(f"No se encontró el archivo: `{EXTRACTED_PATH}`")
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────

    with st.sidebar:
        st.header("Filtros")

        all_cats = sorted({r["category"] for r in extracted})
        cat_options = ["Todas"] + all_cats
        selected_cat = st.selectbox("Categoría", cat_options)

        min_score = st.slider(
            "Score mínimo (similitud semántica)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        )

        if st.button("🎲 Artículo aleatorio"):
            pool = [
                r for r in extracted
                if (selected_cat == "Todas" or r["category"] == selected_cat)
                and r.get("score_pymupdf", 0) >= min_score
                and r.get("score_docling", 0) >= min_score
            ]
            if pool:
                st.session_state["ext_article"] = random.choice(pool)
            else:
                st.warning("No hay artículos que cumplan los filtros.")

    # ── Selección de artículo ─────────────────────────────────────────────────

    # Filtrar corpus según controles
    pool = [
        r for r in extracted
        if (selected_cat == "Todas" or r["category"] == selected_cat)
        and r.get("score_pymupdf", 0) >= min_score
        and r.get("score_docling", 0) >= min_score
    ]

    if not pool:
        st.info("No hay artículos que cumplan los filtros actuales.")
        return

    # Inicializar artículo actual en session_state
    if "ext_article" not in st.session_state or st.session_state["ext_article"] not in pool:
        st.session_state["ext_article"] = pool[0]

    article = st.session_state["ext_article"]

    # ── Encabezado del artículo ───────────────────────────────────────────────

    cat_name = categories.get(article["category"], "")
    st.subheader(article["title"])
    st.caption(
        f"**{article['category']}** — {cat_name} &nbsp;|&nbsp; "
        f"[{article['arxiv_id']}](https://arxiv.org/abs/{article['arxiv_id']})"
    )

    # ── Métricas de calidad ───────────────────────────────────────────────────

    score_pym = article.get("score_pymupdf", 0.0)
    score_doc = article.get("score_docling", 0.0)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(label="Artículos en corpus", value=f"{len(pool):,}")
    col_m2.metric(
        label="Score PyMuPDF",
        value=f"{score_pym:.4f}",
        delta=f"{score_pym - 1.0:.4f}",
        delta_color="normal",
    )
    col_m3.metric(
        label="Score Docling",
        value=f"{score_doc:.4f}",
        delta=f"{score_doc - 1.0:.4f}",
        delta_color="normal",
    )

    # ── Advertencias ─────────────────────────────────────────────────────────

    if extraction_failed(article, "pymupdf"):
        st.error("PyMuPDF: extracción marcada como **failed** en extraction_status.")
    elif score_pym < 0.70:
        st.warning(
            "⚠️ **Extracción PyMuPDF problemática** — posible PDF escaneado o con layout inusual "
            f"(score = {score_pym:.4f})"
        )

    if extraction_failed(article, "docling"):
        st.error("Docling: extracción marcada como **failed** en extraction_status.")
    elif score_doc < 0.70:
        st.warning(
            "⚠️ **Extracción Docling problemática** — posible PDF escaneado o con layout inusual "
            f"(score = {score_doc:.4f})"
        )

    # ── Comparación de abstracts ──────────────────────────────────────────────

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.expander("📄 Abstract API (referencia)", expanded=True):
            st.markdown(article.get("abstract_api") or "_No disponible_")

    with col2:
        with st.expander(f"📑 Abstract PyMuPDF  (score: {score_pym:.4f})", expanded=True):
            text = article.get("abstract_pymupdf", "").strip()
            if text:
                st.markdown(text)
            else:
                st.caption("_Abstract vacío — extracción fallida_")

    with col3:
        with st.expander(f"📑 Abstract Docling  (score: {score_doc:.4f})", expanded=True):
            text = article.get("abstract_docling", "").strip()
            if text:
                st.markdown(text)
            else:
                st.caption("_Abstract vacío — extracción fallida_")

    # ── Navegación manual entre artículos ─────────────────────────────────────

    st.divider()
    idx = pool.index(article) if article in pool else 0
    col_prev, col_info, col_next = st.columns([1, 3, 1])

    with col_prev:
        if st.button("← Anterior") and idx > 0:
            st.session_state["ext_article"] = pool[idx - 1]
            st.rerun()

    with col_info:
        st.caption(f"Artículo {idx + 1} de {len(pool)}")

    with col_next:
        if st.button("Siguiente →") and idx < len(pool) - 1:
            st.session_state["ext_article"] = pool[idx + 1]
            st.rerun()


main()

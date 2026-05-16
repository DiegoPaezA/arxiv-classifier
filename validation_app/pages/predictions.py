"""
Página 2 — Validación de predicciones.

Explora los errores del modelo e identifica patrones por clase y confianza.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
EVAL_DIR    = ROOT / "reports"
EXTRACTED_PATH = ROOT / "data/interim/extracted.json"
CATEGORIES_PATH = ROOT / "configs/categories.json"

SOURCES = ["api", "pymupdf", "docling"]
ITEMS_PER_PAGE = 10


# ── Carga de datos ────────────────────────────────────────────────────────────

@st.cache_data
def load_evaluation(source: str) -> dict | None:
    path = EVAL_DIR / f"evaluation_results_{source}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_abstracts() -> dict[str, dict]:
    """Devuelve {arxiv_id: {abstract_api, abstract_pymupdf, abstract_docling}}."""
    if not EXTRACTED_PATH.exists():
        return {}
    with open(EXTRACTED_PATH) as f:
        data = json.load(f)
    return {
        r["arxiv_id"]: {
            "abstract_api":     r.get("abstract_api", ""),
            "abstract_pymupdf": r.get("abstract_pymupdf", ""),
            "abstract_docling": r.get("abstract_docling", ""),
        }
        for r in data
    }


@st.cache_data
def load_categories() -> dict[str, str]:
    if not CATEGORIES_PATH.exists():
        return {}
    with open(CATEGORIES_PATH) as f:
        cats = json.load(f)
    return {c["code"]: c["name"] for c in cats}


# ── Gráfico de probabilidades ─────────────────────────────────────────────────

def prob_chart(
    probabilities: dict,
    true_label: str | None = None,
    pred_label: str | None = None,
) -> plt.Figure:
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    colors = []
    for lbl in labels:
        if lbl == true_label and lbl == pred_label:
            colors.append("#28a745")  # verde: acierto
        elif lbl == pred_label:
            colors.append("#4e73df")  # azul: predicción incorrecta
        elif lbl == true_label:
            colors.append("#28a745")  # verde: clase real no predicha
        else:
            colors.append("#d3d3d3")  # gris: resto

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.barh(labels, values, color=colors, height=0.6, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Probabilidad", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    return fig


# ── Página ────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🎯 Validación de predicciones")
    st.caption("Explora los errores del modelo y analiza el rendimiento por clase.")

    categories = load_categories()
    abstracts  = load_abstracts()

    # ── Sidebar ───────────────────────────────────────────────────────────────

    with st.sidebar:
        st.header("Filtros")

        source = st.selectbox("Modelo", SOURCES, format_func=lambda s: f"abstract_{s}")

        data = load_evaluation(source)
        if data is None:
            st.warning(f"Archivo no encontrado:\n`reports/evaluation_results_{source}.json`")
            st.info("Ejecuta `src/training/evaluate_model.py --source {source}` primero.")
            return

        label_order = data["label_order"]

        true_cat = st.selectbox(
            "Clase real", ["Todas"] + label_order
        )
        pred_cat = st.selectbox(
            "Clase predicha", ["Todas"] + label_order
        )
        only_errors  = st.checkbox("Solo errores", value=False)
        min_conf     = st.slider("Confianza mínima", 0.0, 1.0, 0.0, 0.05)

    # ── Métricas globales ─────────────────────────────────────────────────────

    m = data["metrics"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy",   f"{m['accuracy']:.4f}")
    col2.metric("F1 macro",   f"{m['f1_macro']:.4f}")
    col3.metric("Artículos evaluados", data["test_articles"])

    # ── Tabla por clase ───────────────────────────────────────────────────────

    with st.expander("📊 Métricas por clase", expanded=True):
        per_class = data["per_class"]
        rows = sorted(
            [
                {
                    "Clase": lbl,
                    "Precision": f"{m['precision']:.4f}",
                    "Recall":    f"{m['recall']:.4f}",
                    "F1":        f"{m['f1']:.4f}",
                    "Support":   m["support"],
                }
                for lbl, m in per_class.items()
            ],
            key=lambda r: float(r["F1"]),
        )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()

    # ── Filtrar predicciones ──────────────────────────────────────────────────

    predictions = data["predictions"]

    filtered = [
        p for p in predictions
        if (true_cat == "Todas" or p["true_label"] == true_cat)
        and (pred_cat == "Todas" or p["predicted_label"] == pred_cat)
        and (not only_errors or not p["correct"])
        and p["confidence"] >= min_conf
    ]

    n_filtered = len(filtered)
    total_pages = max(1, (n_filtered + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)

    # Inicializar y validar página en session_state
    if "pred_page" not in st.session_state:
        st.session_state["pred_page"] = 0
    st.session_state["pred_page"] = min(st.session_state["pred_page"], total_pages - 1)

    current_page = st.session_state["pred_page"]
    page_start   = current_page * ITEMS_PER_PAGE
    page_items   = filtered[page_start : page_start + ITEMS_PER_PAGE]

    st.subheader(f"Predicciones — {n_filtered} resultado(s)")

    if not page_items:
        st.info("No hay predicciones que cumplan los filtros.")
        return

    # ── Cards de predicciones ─────────────────────────────────────────────────

    for pred in page_items:
        aid          = pred["arxiv_id"]
        correct      = pred["correct"]
        true_lbl     = pred["true_label"]
        pred_lbl     = pred["predicted_label"]
        confidence   = pred["confidence"]
        probabilities = pred["probabilities"]

        abstract_key = f"abstract_{source}"
        abstract_text = abstracts.get(aid, {}).get(abstract_key, "_(abstract no disponible)_")

        border_color = "#28a745" if correct else "#dc3545"
        icon = "✅" if correct else "❌"

        with st.container():
            st.markdown(
                f"""<div style="
                    border-left: 5px solid {border_color};
                    padding: 10px 14px;
                    margin-bottom: 4px;
                    background-color: #f8f9fa;
                    border-radius: 4px;
                ">
                <code>{aid}</code> &nbsp;|&nbsp;
                <strong>{true_lbl}</strong> → <strong>{pred_lbl}</strong>
                &nbsp;|&nbsp; confianza: <strong>{confidence:.3f}</strong>
                &nbsp; {icon}
                </div>""",
                unsafe_allow_html=True,
            )

            with st.expander(f"Ver detalle — {aid}", expanded=False):
                st.markdown(f"**Abstract ({source}):** {abstract_text[:300]}...")
                st.markdown(f"[🔗 Ver en arXiv](https://arxiv.org/abs/{aid})")

                fig = prob_chart(probabilities, true_label=true_lbl, pred_label=pred_lbl)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

        st.divider()

    # ── Paginación ────────────────────────────────────────────────────────────

    col_prev, col_info, col_next = st.columns([1, 3, 1])

    with col_prev:
        if st.button("← Anterior", disabled=(current_page == 0)):
            st.session_state["pred_page"] -= 1
            st.rerun()

    with col_info:
        st.caption(
            f"Página {current_page + 1} de {total_pages} "
            f"({page_start + 1}–{min(page_start + ITEMS_PER_PAGE, n_filtered)} "
            f"de {n_filtered})"
        )

    with col_next:
        if st.button("Siguiente →", disabled=(current_page >= total_pages - 1)):
            st.session_state["pred_page"] += 1
            st.rerun()


main()

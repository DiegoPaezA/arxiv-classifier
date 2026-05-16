"""
Página 3 — Clasificar paper nuevo.

Ejecuta el pipeline completo en vivo: API de arXiv → PDF → PyMuPDF
→ similitud semántica → SciBERT → predicción.
También acepta texto libre si no se quiere descargar el PDF.
"""

import json
import re
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR     = ROOT / "reports"
CATEGORIES_PATH = ROOT / "configs/categories.json"

ARXIV_ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_SCHEMA_NS = "{http://arxiv.org/schemas/atom}"

SOURCES = ["api", "pymupdf", "docling"]


# ── Carga de datos y modelos ──────────────────────────────────────────────────

@st.cache_data
def load_categories() -> dict[str, str]:
    if not CATEGORIES_PATH.exists():
        return {}
    with open(CATEGORIES_PATH) as f:
        cats = json.load(f)
    return {c["code"]: c["name"] for c in cats}


@st.cache_data
def load_best_checkpoint(source: str) -> str | None:
    path = REPORTS_DIR / f"best_checkpoint_{source}.json"
    if not path.exists():
        return None
    with open(path) as f:
        info = json.load(f)
    checkpoint = ROOT / info["best_checkpoint"]
    return str(checkpoint) if checkpoint.exists() else None


@st.cache_data
def load_label_order(source: str) -> list[str] | None:
    path = REPORTS_DIR / f"evaluation_results_{source}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["label_order"]


@st.cache_resource
def load_scibert(checkpoint: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_similarity_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── Funciones del pipeline ────────────────────────────────────────────────────

def fetch_arxiv_metadata(arxiv_id: str) -> dict | None:
    """Consulta la API de arXiv y devuelve title, abstract, category, published."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml_bytes = resp.read()
    except Exception as e:
        st.error(f"Error al consultar la API de arXiv: {e}")
        return None

    try:
        root = ET.fromstring(xml_bytes)
        entry = root.find(f"{ARXIV_ATOM_NS}entry")
        if entry is None:
            st.error("La API no devolvió resultados para ese arxiv_id.")
            return None

        title     = (entry.findtext(f"{ARXIV_ATOM_NS}title") or "").strip().replace("\n", " ")
        abstract  = (entry.findtext(f"{ARXIV_ATOM_NS}summary") or "").strip().replace("\n", " ")
        published = (entry.findtext(f"{ARXIV_ATOM_NS}published") or "")[:10]

        # Categoría primaria
        prim = entry.find(f"{ARXIV_SCHEMA_NS}primary_category")
        category = prim.get("term", "") if prim is not None else ""

        return {"title": title, "abstract": abstract, "category": category, "published": published}
    except ET.ParseError as e:
        st.error(f"Error al parsear la respuesta XML de arXiv: {e}")
        return None


def download_pdf(arxiv_id: str, dest: Path) -> bool:
    """Descarga el PDF a dest. Devuelve True si tuvo éxito."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (arxiv-classifier research project)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as e:
        st.warning(f"No se pudo descargar el PDF: {e}")
        return False


def extract_abstract_pymupdf(pdf_path: Path) -> str:
    """Extrae el abstract del PDF con PyMuPDF usando heurística de sección."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        text = ""
        for i in range(min(2, len(doc))):
            text += doc[i].get_text()
        doc.close()

        # Buscar la sección Abstract (ignora mayúsculas/minúsculas)
        match = re.search(r"\bAbstract\b|\bABSTRACT\b", text)
        if match:
            rest = text[match.end():]
            # Cortar en la siguiente sección: encabezado numerado o en mayúsculas
            next_sec = re.search(
                r"\n\s*(?:\d[\s\.\)]+[A-Z]|Introduction|INTRODUCTION|Keywords|KEYWORDS)",
                rest,
            )
            end = next_sec.start() if next_sec else min(1800, len(rest))
            abstract = rest[:end]
        else:
            abstract = text[:1500]

        # Limpiar saltos de línea y espacios múltiples
        abstract = re.sub(r"[\r\n]+", " ", abstract)
        abstract = re.sub(r" {2,}", " ", abstract).strip()
        return abstract

    except Exception as e:
        st.warning(f"PyMuPDF no pudo procesar el PDF: {e}")
        return ""


def compute_similarity(text1: str, text2: str, sim_model) -> float:
    from sentence_transformers import util
    e1 = sim_model.encode(text1, convert_to_tensor=True)
    e2 = sim_model.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(e1, e2).item())


def classify_text(
    text: str,
    tokenizer,
    model,
    label_order: list[str],
) -> dict[str, float]:
    """Tokeniza el texto, hace inferencia y devuelve probabilidades por clase."""
    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    return {label: float(prob) for label, prob in zip(label_order, probs)}


# ── Visualización de resultados ───────────────────────────────────────────────

def show_result_chart(
    probabilities: dict[str, float],
    true_label: str | None,
    pred_label: str,
    categories: dict[str, str],
) -> None:
    """Gráfico horizontal con barras coloreadas: verde=clase real, azul=predicha."""
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    colors = []
    for lbl in labels:
        if lbl == true_label and lbl == pred_label:
            colors.append("#28a745")
        elif lbl == pred_label:
            colors.append("#4e73df")
        elif lbl == true_label:
            colors.append("#28a745")
        else:
            colors.append("#d3d3d3")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Probabilidad")
    ax.set_title("Distribución de probabilidades por clase")

    # Etiquetas de valor en cada barra
    for bar, val in zip(bars, values):
        if val > 0.01:
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center", fontsize=8,
            )

    # Leyenda de colores
    legend_items = [
        plt.Rectangle((0, 0), 1, 1, fc="#28a745", label="Clase real"),
        plt.Rectangle((0, 0), 1, 1, fc="#4e73df", label="Clase predicha"),
        plt.Rectangle((0, 0), 1, 1, fc="#d3d3d3", label="Resto"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Top 3 en tabla
    top3 = sorted_items[:3]
    st.markdown("**Top 3 predicciones:**")
    rows = [
        {
            "Categoría": lbl,
            "Probabilidad": f"{prob:.4f}",
            "Descripción": categories.get(lbl, "—"),
        }
        for lbl, prob in top3
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ── Página ────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🤖 Clasificar paper nuevo")
    st.caption("Clasifica un paper que no está en el dataset usando el pipeline completo.")

    categories = load_categories()

    # ── Sidebar ───────────────────────────────────────────────────────────────

    with st.sidebar:
        st.header("Configuración")
        source = st.selectbox("Modelo", SOURCES, format_func=lambda s: f"abstract_{s}")
        input_mode = st.radio("Modo de entrada", ["arxiv_id", "Texto libre"], index=0)

    # Cargar modelo seleccionado
    checkpoint = load_best_checkpoint(source)
    label_order = load_label_order(source)

    if checkpoint is None:
        st.error(
            f"Modelo no encontrado para `abstract_{source}`. "
            f"Ejecuta primero:\n\n"
            f"`uv run src/training/train_bert.py --source {source}`"
        )
        return

    if label_order is None:
        # Fallback: intentar leer del propio modelo
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(checkpoint)
            label_order = [cfg.id2label[i] for i in sorted(cfg.id2label)]
        except Exception:
            label_order = list(categories.keys())

    # ── Modo texto libre ──────────────────────────────────────────────────────

    if input_mode == "Texto libre":
        st.subheader("Clasificación por texto libre")
        text_input = st.text_area(
            "Pega el abstract del paper aquí",
            height=200,
            placeholder="Abstract en inglés...",
        )
        if st.button("Clasificar", key="btn_free_text") and text_input.strip():
            try:
                tokenizer, model = load_scibert(checkpoint)
            except Exception as e:
                st.error(f"No se pudo cargar el modelo: {e}")
                return

            with st.spinner("Clasificando..."):
                probabilities = classify_text(text_input, tokenizer, model, label_order)

            pred_label = max(probabilities, key=probabilities.get)
            confidence = probabilities[pred_label]

            st.success(
                f"**Categoría predicha: `{pred_label}`** — "
                f"{categories.get(pred_label, '')} (confianza: {confidence:.3f})"
            )
            show_result_chart(probabilities, None, pred_label, categories)
            st.session_state["clf_result"] = probabilities

        return

    # ── Modo arxiv_id ─────────────────────────────────────────────────────────

    st.subheader("Clasificación por arxiv_id")
    arxiv_id = st.text_input("arxiv_id", placeholder="Ej: 2301.07041")

    if not st.button("🚀 Clasificar paper", key="btn_classify"):
        st.info("Ingresa un arxiv_id y pulsa **Clasificar paper** para ejecutar el pipeline.")
        return

    if not arxiv_id.strip():
        st.warning("Por favor ingresa un arxiv_id.")
        return

    arxiv_id = arxiv_id.strip()
    st.divider()

    # ── Paso 1: Metadata de la API ────────────────────────────────────────────

    with st.status("Paso 1 — Obteniendo metadata de la API de arXiv..."):
        metadata = fetch_arxiv_metadata(arxiv_id)

    if metadata is None:
        return

    st.markdown(f"**{metadata['title']}**")
    st.caption(
        f"Categoría oficial arXiv: `{metadata['category']}` "
        f"| Publicado: {metadata['published']} "
        f"| [Ver en arXiv](https://arxiv.org/abs/{arxiv_id})"
    )

    abstract_api = metadata["abstract"]
    official_category = metadata["category"]
    abstract_for_classification = abstract_api  # default

    # ── Paso 2 + 3: Descarga y extracción (solo para pymupdf) ────────────────

    abstract_pymupdf = ""

    if source == "pymupdf":
        with st.status("Paso 2 — Descargando PDF..."):
            tmp_dir = Path(tempfile.gettempdir())
            pdf_path = tmp_dir / f"{arxiv_id}.pdf"
            ok = download_pdf(arxiv_id, pdf_path)

        if ok:
            with st.status("Paso 3 — Extrayendo abstract con PyMuPDF..."):
                abstract_pymupdf = extract_abstract_pymupdf(pdf_path)
                # Limpiar archivo temporal
                try:
                    pdf_path.unlink(missing_ok=True)
                except Exception:
                    pass

            if abstract_pymupdf:
                abstract_for_classification = abstract_pymupdf
            else:
                st.warning("PyMuPDF no pudo extraer el abstract. Usando abstract de la API como fallback.")
                abstract_for_classification = abstract_api
        else:
            st.warning("No se pudo descargar el PDF. Usando abstract de la API como fallback.")
            abstract_for_classification = abstract_api

    elif source == "docling":
        st.warning(
            "Docling requiere conversión local del PDF (no disponible en modo online). "
            "Se usará el **abstract de la API** como texto de entrada para el modelo `abstract_docling`."
        )
        abstract_for_classification = abstract_api

    # ── Paso 4: Comparación de abstracts ─────────────────────────────────────

    show_comparison = source == "pymupdf" and abstract_pymupdf

    if show_comparison:
        st.subheader("Paso 4 — Comparación de abstracts")
        col_api, col_pym = st.columns(2)

        with col_api:
            st.markdown("**Abstract oficial (API)**")
            st.markdown(abstract_api)

        with col_pym:
            st.markdown("**Abstract extraído (PyMuPDF)**")
            st.markdown(abstract_pymupdf)

        with st.spinner("Calculando similitud semántica..."):
            sim_model = load_similarity_model()
            sim_score = compute_similarity(abstract_api, abstract_pymupdf, sim_model)

        st.metric(
            label="Similitud semántica (MiniLM)",
            value=f"{sim_score:.4f}",
            delta=f"{sim_score - 1.0:.4f}",
            delta_color="normal",
        )
        if sim_score < 0.70:
            st.warning("Similitud baja — la extracción PyMuPDF puede contener ruido significativo.")
        st.divider()

    # ── Paso 5: Clasificación ─────────────────────────────────────────────────

    st.subheader("Paso 5 — Clasificación con SciBERT")
    st.caption(f"Modelo: `models/scibert_{source}/best_model/` · Texto usado: `abstract_{source}`")

    try:
        tokenizer, model = load_scibert(checkpoint)
    except Exception as e:
        st.error(f"No se pudo cargar el modelo SciBERT: {e}")
        return

    with st.spinner("Clasificando..."):
        probabilities = classify_text(
            abstract_for_classification, tokenizer, model, label_order
        )

    pred_label = max(probabilities, key=probabilities.get)
    confidence = probabilities[pred_label]
    true_label = official_category if official_category in label_order else None

    st.divider()

    # ── Paso 6: Resultado ─────────────────────────────────────────────────────

    st.subheader("Paso 6 — Resultado")

    acierto = true_label is not None and pred_label == true_label

    if true_label is None:
        st.info(
            f"**Categoría predicha: `{pred_label}`** — "
            f"{categories.get(pred_label, '')} (confianza: {confidence:.3f})\n\n"
            f"_(La categoría oficial `{official_category}` no está en el conjunto de 10 clases del modelo)_"
        )
    elif acierto:
        st.success(
            f"✅ **Predicción correcta: `{pred_label}`** — "
            f"{categories.get(pred_label, '')} (confianza: {confidence:.3f})\n\n"
            f"Categoría oficial arXiv: `{official_category}`"
        )
    else:
        st.error(
            f"❌ **Predicción incorrecta: `{pred_label}`** — "
            f"{categories.get(pred_label, '')} (confianza: {confidence:.3f})\n\n"
            f"Categoría oficial arXiv: `{official_category}`"
        )

    show_result_chart(probabilities, true_label, pred_label, categories)
    st.session_state["clf_result"] = probabilities


main()

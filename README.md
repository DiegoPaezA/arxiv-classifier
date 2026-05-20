# Clasificación multiclase de papers de arXiv con SciBERT

Pipeline completo de ciencia de datos que descarga papers de arXiv,
extrae abstracts desde PDF con PyMuPDF y Docling, entrena SciBERT
en fine-tuning para clasificación en 10 categorías de cs.*, y evalúa
el impacto de la calidad de extracción en el rendimiento del modelo.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-orange)

---

## Resultados

| Fuente            | F1 macro | Accuracy | Artículos |
|-------------------|----------|----------|-----------|
| abstract_api      |  0.7519  |  0.7667  |    300    |
| abstract_pymupdf  |  0.7722  |  0.7867  |    300    |
| abstract_docling  |  0.7521  |  0.7710  |    297    |

---

## Requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — gestor de entorno y dependencias

For linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- GPU recomendada para entrenamiento (6 GB+ VRAM).
  CPU funciona pero el entrenamiento es significativamente más lento.
- ~10 GB de espacio en disco para PDFs y modelos.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd arxiv_classifier

# 2. Instalar dependencias
uv sync

# 3. Crear archivo de variables de entorno
echo "HF_TOKEN=<tu_token>" > .env

# 4. Verificar instalación
uv run python -c "import torch; print(torch.cuda.is_available())"
```

---

## Pipeline completo

### Etapa 1 — Explorar categorías

Consulta la API de arXiv para listar y filtrar categorías disponibles.

```bash
uv run python src/collect/explore_categories.py
```

**Salida:** `configs/categories.json`, `reports/category_exploration.csv`

```bash
# Verificación
cat configs/categories.json
```

---

### Etapa 2 — Descargar metadata

Descarga los metadatos (título, abstract, autores) de los artículos seleccionados.

```bash
uv run python src/collect/download_metadata.py
```

**Salida:** `data/raw/metadata.json`

```bash
# Verificación — artículos por categoría
python -c "
import json
from collections import Counter
data = json.load(open('data/raw/metadata.json'))
print(Counter(d['category'] for d in data))
"
```

---

### Etapa 3 — Descargar PDFs

Descarga los PDFs de los artículos. Incluye un delay de 2 s entre requests para respetar los límites de la API.

> **Advertencia:** ~4–8 GB de descarga, puede tardar 2–3 horas.

```bash
uv run python src/collect/download_pdfs.py
```

**Salida:** `data/pdfs/{categoria}/{arxiv_id}.pdf`

```bash
# Verificación
find data/pdfs -name "*.pdf" | wc -l
```

---

### Etapa 4 — Extraer abstracts

Extrae abstracts desde los PDFs con dos métodos diferentes. Docling es más lento (~2–3 horas en CPU).

```bash
uv run python src/extraction/extract_pymupdf.py
uv run python src/extraction/extract_docling.py
```

**Salida:**
- `data/interim/extracted_pymupdf.json`
- `data/interim/extracted_docling.json`

```bash
# Verificación — tasa de éxito por extractor
python -c "
import json
for source in ['pymupdf', 'docling']:
    data = json.load(open(f'data/interim/extracted_{source}.json'))
    ok = sum(1 for d in data if d.get('abstract'))
    print(f'{source}: {ok}/{len(data)} ({ok/len(data):.1%})')
"
```

---

### Etapa 5 — Comparar extracciones

Calcula similitud semántica entre los abstracts de la API y los extraídos desde PDF para seleccionar la mejor extracción por artículo.

> Requiere `sentence-transformers`. Puede tardar 10–15 min en CPU.

```bash
uv run python src/extraction/compare_abstracts.py
```

**Salida:**
- `data/interim/extracted.json`
- `data/interim/embeddings/*.npy`

```bash
# Verificación — scores promedio por extractor
python -c "
import json
data = json.load(open('data/interim/extracted.json'))
for src in ['pymupdf', 'docling']:
    scores = [d[f'score_{src}'] for d in data if f'score_{src}' in d]
    if scores:
        print(f'{src}: {sum(scores)/len(scores):.4f}')
"
```

---

### Etapa 6 — Construir dataset

Consolida las extracciones en un dataset con las tres fuentes de texto y genera los splits train/val/test.

```bash
uv run python src/dataset/build_dataset.py
uv run python src/dataset/split_dataset.py
```

**Salida:**
- `data/processed/dataset.json`
- `data/processed/train.json`
- `data/processed/val.json`
- `data/processed/test.json`

```bash
# Verificación — balance de clases en train
python -c "
import json
from collections import Counter
data = json.load(open('data/processed/train.json'))
print(Counter(d['label'] for d in data))
"
```

---

### Etapa 7 — Entrenar modelos

Fine-tuning de SciBERT con cada una de las tres fuentes de texto. Ejecutar en orden.

> ~30–40 min por modelo en GPU de 6 GB.

```bash
uv run python src/training/train_bert.py --source api
uv run python src/training/train_bert.py --source pymupdf
uv run python src/training/train_bert.py --source docling
```

**Argumentos opcionales:**
- `--epochs N` — número de epochs (default: 3)
- `--test` — smoke test rápido (1 epoch, 50 artículos)

**Salida:** `models/scibert_{source}/best_model/`

```bash
# Verificación — historial de entrenamiento
python -c "
import json, glob
for f in sorted(glob.glob('models/*/training_history.json')):
    h = json.load(open(f))
    print(f, '— val_f1:', h[-1].get('val_f1'))
"
```

---

### Etapa 8 — Evaluar modelos

Evalúa cada modelo en el conjunto de test y genera el reporte completo.

```bash
uv run python src/training/evaluate_model.py --source api
uv run python src/training/evaluate_model.py --source pymupdf
uv run python src/training/evaluate_model.py --source docling
```

**Salida:** `reports/evaluation_results_{source}.json`

```bash
# Verificación — F1 macro de los 3 modelos
python -c "
import json, glob
for f in sorted(glob.glob('reports/evaluation_results_*.json')):
    r = json.load(open(f))
    print(f.split('_')[-1].replace('.json',''), '— F1:', r.get('f1_macro'))
"
```

---

## Notebooks

Abrir Jupyter:

```bash
uv run jupyter notebook notebooks/
```

| # | Notebook | Descripción | Requiere |
|---|----------|-------------|----------|
| 01 | project_overview | Introducción y exploración de categorías | `categories.json` |
| 02 | pdf_extraction | PyMuPDF vs Docling, heurísticas de limpieza | `PDFs` + `extracted_pymupdf.json` |
| 03 | semantic_similarity | Embeddings y comparación de calidad de extracción | `extracted.json` + `embeddings/` |
| 04 | build_dataset | Dataset final, splits y distribución de clases | `dataset.json` |
| 05 | finetune_bert | SciBERT, fine-tuning y curvas de aprendizaje | `training_history_*.json` |
| 06 | model_errors | Análisis de errores, matrices de confusión | `evaluation_results_*.json` |
| 07 | comparison | Resumen comparativo del proyecto | todos los anteriores |

---

## Miniapp de validación

```bash
uv run streamlit run validation_app/app.py
```

### Página 1 — Validación de extracción

Compara los tres abstracts lado a lado (API, PyMuPDF, Docling) con sus scores de similitud semántica.

[screenshot]

### Página 2 — Validación de predicciones

Explora errores del modelo con filtros por clase y nivel de confianza.

[screenshot]

### Página 3 — Clasificar paper nuevo

Ingresa un `arxiv_id` o texto libre y obtén una predicción en tiempo real con los tres modelos.

[screenshot]

---

## Estructura del proyecto

```
arxiv_classifier/
├── configs/                    ← categorías y mapeo de etiquetas
│   ├── categories_candidates.json
│   ├── categories.json
│   └── label_map.json
├── data/
│   ├── pdfs/                   ← 2,000 PDFs organizados por categoría
│   ├── raw/                    ← metadata.json, download_report.json
│   ├── interim/                ← extracted.json, embeddings/
│   └── processed/              ← dataset.json, train/val/test.json
├── src/
│   ├── collect/                ← descarga de metadata y PDFs
│   │   ├── explore_categories.py
│   │   ├── download_metadata.py
│   │   └── download_pdfs.py
│   ├── extraction/             ← extracción y comparación de abstracts
│   │   ├── extract_pymupdf.py
│   │   ├── extract_docling.py
│   │   └── compare_abstracts.py
│   ├── dataset/                ← construcción y split del dataset
│   │   ├── build_dataset.py
│   │   └── split_dataset.py
│   ├── training/               ← fine-tuning y evaluación de SciBERT
│   │   ├── train_bert.py
│   │   └── evaluate_model.py
│   └── utils/                  ← utilidades compartidas
│       ├── config.py
│       └── text_utils.py
├── notebooks/                  ← análisis y visualizaciones
├── validation_app/             ← miniapp Streamlit de validación
│   ├── app.py
│   └── pages/
│       ├── extraction.py
│       ├── predictions.py
│       └── classifier.py
├── models/                     ← modelos entrenados (no versionados)
│   ├── scibert_api/best_model/
│   ├── scibert_pymupdf/best_model/
│   └── scibert_docling/best_model/
├── reports/                    ← métricas y resultados de evaluación
├── .env                        ← variables de entorno (no versionar)
├── pyproject.toml
└── README.md
```

---

## Categorías del dataset

| Código | Nombre | Artículos |
|--------|--------|-----------|
| cs.AI | Artificial Intelligence | 200 |
| cs.CL | Computation and Language | 200 |
| cs.CR | Cryptography and Security | 200 |
| cs.CV | Computer Vision | 200 |
| cs.DB | Databases | 200 |
| cs.IR | Information Retrieval | 200 |
| cs.LG | Machine Learning | 200 |
| cs.NE | Neural and Evolutionary Computing | 200 |
| cs.RO | Robotics | 200 |
| cs.SE | Software Engineering | 200 |

> **Nota sobre solapamiento semántico:** `cs.AI`, `cs.LG` y `cs.CL` comparten vocabulario de forma significativa. Esto es intencional: refleja un problema real de clasificación donde las fronteras entre categorías no son perfectamente nítidas.

---

## Notas para reproducibilidad

- Todos los scripts usan `seed=42`.
- La API de arXiv puede devolver artículos diferentes según la fecha de ejecución — los resultados pueden variar ligeramente.
- Los PDFs se descargan con un delay de 2 segundos entre requests para respetar los límites de la API de arXiv.
- El archivo `.env` nunca debe subirse al repositorio (está en `.gitignore`).
- Para reproducir exactamente los resultados, usar los archivos en `data/processed/` sin regenerar el dataset.

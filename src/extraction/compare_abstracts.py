import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_json_by_id(path: Path) -> dict:
    """Carga JSON y retorna dict keyed por arxiv_id."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item.get("arxiv_id"): item for item in data}


def get_abstract(item: dict, key: str) -> str:
    """Extrae abstract de un item, retorna string vacío si no existe."""
    if item is None:
        return ""
    abstract = item.get(key, "") or ""
    return abstract.strip()


def compute_similarity(text1: str, text2: str, embedding_model, embeddings_cache: dict) -> float:
    """Calcula cosine similarity entre dos textos."""
    if not text1 or not text2:
        return 0.0

    # Usar cache para evitar re-encodificar
    if text1 not in embeddings_cache:
        embeddings_cache[text1] = embedding_model.encode(text1)
    if text2 not in embeddings_cache:
        embeddings_cache[text2] = embedding_model.encode(text2)

    emb1 = embeddings_cache[text1]
    emb2 = embeddings_cache[text2]

    # cosine_similarity retorna matriz 2D
    sim = cosine_similarity([emb1], [emb2])[0][0]
    return float(sim)


def encode_texts_batch(texts: list, model) -> np.ndarray:
    """Codifica textos en lotes de 64."""
    batch_size = 64
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings) if embeddings else np.array([])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara abstracts usando embeddings de sentence-transformers."
    )
    parser.add_argument("--test", action="store_true", help="Procesa solo 10 artículos para prueba")
    args = parser.parse_args()

    metadata_path = Path("data/raw/metadata.json")
    pymupdf_path = Path("data/interim/extracted_pymupdf.json")
    docling_path = Path("data/interim/extracted_docling.json")
    output_path = Path("data/interim/extracted.json")
    embeddings_dir = Path("data/interim/embeddings")

    # Validar archivos de entrada
    if not metadata_path.exists():
        sys.exit(f"Error: {metadata_path} no existe")

    # Cargar datos
    print("Cargando datos...")
    metadata = load_json_by_id(metadata_path)
    pymupdf_data = load_json_by_id(pymupdf_path)
    docling_data = load_json_by_id(docling_path)

    if not metadata:
        sys.exit("Error: metadata.json está vacío")

    # Obtener list de arxiv_ids a procesar
    arxiv_ids = list(metadata.keys())
    if args.test:
        arxiv_ids = arxiv_ids[:10]
        print(f"Modo TEST: procesando solo {len(arxiv_ids)} artículos\n")

    print(f"Total a procesar: {len(arxiv_ids)} artículos")
    print("Inicializando sentence-transformer con 'all-MiniLM-L6-v2'...")

    # Cargar modelo
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Modelo cargado. Dimensionalidad: {model.get_embedding_dimension()}\n")

    # Preparar datos para embeddings
    print("Preparando textos para embeddings...")
    abstracts_api = []
    abstracts_pymupdf = []
    abstracts_docling = []
    results = []

    for arxiv_id in arxiv_ids:
        meta_item = metadata[arxiv_id]
        pymupdf_item = pymupdf_data.get(arxiv_id)
        docling_item = docling_data.get(arxiv_id)

        # Extraer abstracts
        abstract_api = get_abstract(meta_item, "abstract_api")
        abstract_pymupdf = get_abstract(pymupdf_item, "abstract_pymupdf")
        abstract_docling = get_abstract(docling_item, "abstract_docling")

        abstracts_api.append(abstract_api)
        abstracts_pymupdf.append(abstract_pymupdf)
        abstracts_docling.append(abstract_docling)

        # Crear estructura de resultado
        result = {
            "arxiv_id": arxiv_id,
            "category": meta_item.get("primary_category", "unknown"),
            "title": meta_item.get("title", ""),
            "abstract_api": abstract_api,
            "abstract_pymupdf": abstract_pymupdf,
            "abstract_docling": abstract_docling,
            "score_pymupdf": 0.0,
            "score_docling": 0.0,
            "extraction_status": {
                "pymupdf": "ok" if abstract_pymupdf else "failed",
                "docling": "ok" if abstract_docling else "failed",
            },
        }
        results.append(result)

    # Generar embeddings
    print("Generando embeddings...")

    # Usar vectores de ceros para abstracts vacíos
    embeddings_api = encode_texts_batch(abstracts_api, model)
    embeddings_pymupdf = encode_texts_batch(abstracts_pymupdf, model)
    embeddings_docling = encode_texts_batch(abstracts_docling, model)

    # Para mantener el orden, reemplazar embeddings de textos vacíos con ceros
    for i, (text_api, text_pymupdf, text_docling) in enumerate(
        zip(abstracts_api, abstracts_pymupdf, abstracts_docling)
    ):
        if not text_api:
            embeddings_api[i] = np.zeros(embeddings_api.shape[1])
        if not text_pymupdf:
            embeddings_pymupdf[i] = np.zeros(embeddings_pymupdf.shape[1])
        if not text_docling:
            embeddings_docling[i] = np.zeros(embeddings_docling.shape[1])

    print("Calculando similitudes...")

    # Calcular similitudes
    for i in range(len(results)):
        emb_api = embeddings_api[i].reshape(1, -1)
        emb_pymupdf = embeddings_pymupdf[i].reshape(1, -1)
        emb_docling = embeddings_docling[i].reshape(1, -1)

        # cosine_similarity retorna matriz (1, 1)
        score_pymupdf = (
            float(cosine_similarity(emb_api, emb_pymupdf)[0][0])
            if abstracts_pymupdf[i]
            else 0.0
        )
        score_docling = (
            float(cosine_similarity(emb_api, emb_docling)[0][0])
            if abstracts_docling[i]
            else 0.0
        )

        results[i]["score_pymupdf"] = score_pymupdf
        results[i]["score_docling"] = score_docling

    # Guardar extracted.json
    print(f"Guardando {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Guardar embeddings
    print(f"Guardando embeddings en {embeddings_dir}...")
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_dir / "abstract_api.npy", embeddings_api)
    np.save(embeddings_dir / "abstract_pymupdf.npy", embeddings_pymupdf)
    np.save(embeddings_dir / "abstract_docling.npy", embeddings_docling)

    # Guardar lista de arxiv_ids en el mismo orden
    arxiv_ids_json = embeddings_dir / "arxiv_ids.json"
    arxiv_ids_json.write_text(json.dumps(arxiv_ids, indent=2), encoding="utf-8")

    # Generar tabla de resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS")
    print("=" * 80)

    # Estadísticas por librería
    scores_pymupdf = [r["score_pymupdf"] for r in results if r["score_pymupdf"] > 0]
    scores_docling = [r["score_docling"] for r in results if r["score_docling"] > 0]

    def print_stats(name: str, scores: list, results_data: list) -> None:
        if not scores:
            print(f"\n{name}:")
            print("  No hay datos para comparar (todos los abstracts están vacíos)")
            return

        scores_arr = np.array(scores)
        all_scores = [r[f"score_{name.lower()}"] for r in results_data]

        print(f"\n{name}:")
        print(f"  Score promedio:  {np.mean(all_scores):.4f}")
        print(f"  Score mediano:   {np.median(all_scores):.4f}")
        print(f"  % score > 0.90:  {len([s for s in all_scores if s > 0.90]) / len(all_scores) * 100:.1f}%")
        print(f"  % score < 0.70:  {len([s for s in all_scores if s < 0.70]) / len(all_scores) * 100:.1f}%")

    # Extraer parte del nombre para comparación
    score_key_pymupdf = "score_pymupdf"
    score_key_docling = "score_docling"

    all_scores_pymupdf = [r[score_key_pymupdf] for r in results]
    all_scores_docling = [r[score_key_docling] for r in results]

    print("\nPyMuPDF:")
    if all_scores_pymupdf:
        pymupdf_arr = np.array(all_scores_pymupdf)
        print(f"  Score promedio:  {np.mean(all_scores_pymupdf):.4f}")
        print(f"  Score mediano:   {np.median(all_scores_pymupdf):.4f}")
        print(f"  % score > 0.90:  {len([s for s in all_scores_pymupdf if s > 0.90]) / len(all_scores_pymupdf) * 100:.1f}%")
        print(f"  % score < 0.70:  {len([s for s in all_scores_pymupdf if s < 0.70]) / len(all_scores_pymupdf) * 100:.1f}%")

    print("\nDocling:")
    if all_scores_docling:
        docling_arr = np.array(all_scores_docling)
        print(f"  Score promedio:  {np.mean(all_scores_docling):.4f}")
        print(f"  Score mediano:   {np.median(all_scores_docling):.4f}")
        print(f"  % score > 0.90:  {len([s for s in all_scores_docling if s > 0.90]) / len(all_scores_docling) * 100:.1f}%")
        print(f"  % score < 0.70:  {len([s for s in all_scores_docling if s < 0.70]) / len(all_scores_docling) * 100:.1f}%")

    # Desglose por categoría
    print("\nDesglose por categoría:")
    by_category = defaultdict(list)
    for result in results:
        category = result["category"]
        by_category[category].append(result)

    for category in sorted(by_category.keys()):
        cat_results = by_category[category]
        cat_scores_pymupdf = [r["score_pymupdf"] for r in cat_results]
        cat_scores_docling = [r["score_docling"] for r in cat_results]

        print(f"\n  {category} ({len(cat_results)} artículos):")
        print(
            f"    PyMuPDF promedio: {np.mean(cat_scores_pymupdf):.4f} | "
            f"Docling promedio: {np.mean(cat_scores_docling):.4f}"
        )

    print("\n" + "=" * 80)
    if args.test:
        print(f"Test completado: {len(results)} artículos procesados")
        print(f"Verificar {output_path} y {embeddings_dir} antes de continuar")
    else:
        print(f"Extractos guardados: {output_path}")
        print(f"Embeddings guardados: {embeddings_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

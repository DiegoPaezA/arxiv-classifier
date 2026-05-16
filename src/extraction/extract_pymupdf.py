import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

try:
    import fitz
except ImportError:
    sys.exit("Error: pymupdf no está instalado. Corre: uv add pymupdf")


def append_to_json(result: dict, json_path: Path) -> None:
    """Agrega un resultado al archivo JSON de forma atómica."""
    json_path = Path(json_path)
    tmp_path = json_path.parent / (json_path.name + ".tmp")

    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        data = []

    data.append(result)

    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, json_path)


def extract_abstract_from_text(text: str) -> tuple[str, str | None]:
    """
    Extrae el abstract del texto del PDF.
    Retorna (abstract, extraction_method).
    extraction_method: "pattern_based", "fallback", o None si falla.
    """
    # Buscar patrón "Abstract" con separador opcional
    match = re.search(
        r"(?:^|\n)\s*abstract\s*(?:[:—\-])?\s*\n(.*?)(?=\n\s*(?:introduction|keywords|references|[0-9]+\.|conclusion|related|acknowledgment)|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if match:
        abstract = match.group(1).strip()
        if abstract:
            return abstract, "pattern_based"

    # Fallback: primeros 1500 caracteres
    abstract = text[:1500].strip()
    if abstract:
        return abstract, "fallback"

    return "", None


def clean_text(text: str) -> str:
    """Limpia y normaliza el texto extraído."""
    # Unir líneas partidas por guión
    text = re.sub(r"-\n", "", text)
    # Reemplazar saltos de línea con espacios
    text = re.sub(r"\n", " ", text)
    # Normalizar espacios múltiples
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_title_from_text(text: str) -> str:
    """Intenta extraer el título del PDF (primeras líneas antes de Abstract)."""
    lines = text.split("\n")
    title_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Parar cuando llegamos a Abstract
        if stripped.lower().startswith("abstract"):
            break
        title_lines.append(stripped)
        # Limitar a primeras líneas
        if len(title_lines) >= 5:
            break

    # Tomar el texto más largo de las primeras líneas como probable título
    if title_lines:
        return " ".join(title_lines[:2]).strip()
    return ""


def save_raw_text(arxiv_id: str, text: str, raw_dir: Path) -> str | None:
    """
    Guarda el texto raw si no existe.
    Retorna ruta del archivo si fue guardado, None si falló.
    """
    raw_path = raw_dir / f"{arxiv_id}.txt"

    # Si ya existe, no regenerar
    if raw_path.exists():
        return str(raw_path)

    try:
        raw_path.write_text(text, encoding="utf-8")
        return str(raw_path)
    except Exception as e:
        # Registrar error pero no detener el proceso
        print(f"    [AVISO] Error guardando raw para {arxiv_id}: {e}")
        return None


def process_pdf(pdf_path: Path, raw_dir: Path) -> dict:
    """Procesa un PDF y extrae el abstract y título, midiendo tiempos."""
    try:
        # Medir extracción de texto
        tiempo_inicio_extraccion = time.perf_counter()
        doc = fitz.open(pdf_path)
        # Extraer texto de primeras 2 páginas
        text = ""
        for page_num in range(min(2, len(doc))):
            text += doc[page_num].get_text()
        doc.close()
        tiempo_extraccion_texto = round(time.perf_counter() - tiempo_inicio_extraccion, 4)

        if not text.strip():
            return {
                "title": "",
                "abstract_pymupdf": "",
                "extraction_method": None,
                "raw_path": None,
                "status": "failed",
                "timing": {
                    "extraccion_texto_seg": tiempo_extraccion_texto,
                    "procesamiento_seg": 0.0,
                    "total_seg": tiempo_extraccion_texto,
                },
            }

        # Medir procesamiento (guardar raw, extraer y limpiar)
        tiempo_inicio_procesamiento = time.perf_counter()

        # Guardar texto raw antes de procesar
        arxiv_id = pdf_path.stem
        raw_path = save_raw_text(arxiv_id, text, raw_dir)

        # Extraer y limpiar abstract
        abstract, method = extract_abstract_from_text(text)
        abstract = clean_text(abstract)

        # Extraer título
        title = extract_title_from_text(text)
        title = clean_text(title)

        tiempo_procesamiento = round(time.perf_counter() - tiempo_inicio_procesamiento, 4)
        tiempo_total = round(tiempo_extraccion_texto + tiempo_procesamiento, 4)

        return {
            "title": title,
            "abstract_pymupdf": abstract,
            "extraction_method": method,
            "raw_path": raw_path,
            "status": "success",
            "timing": {
                "extraccion_texto_seg": tiempo_extraccion_texto,
                "procesamiento_seg": tiempo_procesamiento,
                "total_seg": tiempo_total,
            },
        }
    except Exception as e:
        return {
            "title": "",
            "abstract_pymupdf": "",
            "extraction_method": None,
            "raw_path": None,
            "status": "failed",
            "timing": {
                "extraccion_texto_seg": 0.0,
                "procesamiento_seg": 0.0,
                "total_seg": 0.0,
            },
        }


def generate_timing_report(results: list, output_dir: Path) -> None:
    """Genera un reporte con estadísticas de timing."""
    if not results:
        return

    # Recolectar tiempos totales y por categoría
    tiempos_totales = []
    por_categoria = defaultdict(lambda: {"tiempos": [], "total_seg": 0.0})

    for result in results:
        if result.get("status") == "success":
            tiempo_total = result.get("timing", {}).get("total_seg", 0.0)
            tiempos_totales.append(tiempo_total)

            categoria = result.get("category", "unknown")
            por_categoria[categoria]["tiempos"].append(tiempo_total)

    if not tiempos_totales:
        return

    # Calcular estadísticas generales
    tiempo_total_acumulado = round(sum(tiempos_totales), 4)
    tiempo_promedio = round(tiempo_total_acumulado / len(tiempos_totales), 4)
    tiempo_mediano = round(median(tiempos_totales), 4)
    tiempo_min = round(min(tiempos_totales), 4)
    tiempo_max = round(max(tiempos_totales), 4)

    # Calcular percentil 95
    tiempos_sorted = sorted(tiempos_totales)
    idx_p95 = int(len(tiempos_sorted) * 0.95)
    tiempo_p95 = round(tiempos_sorted[idx_p95], 4) if idx_p95 < len(tiempos_sorted) else 0.0

    # Calcular por categoría
    por_categoria_stats = {}
    for cat, data in por_categoria.items():
        if data["tiempos"]:
            promedio = round(sum(data["tiempos"]) / len(data["tiempos"]), 4)
            total = round(sum(data["tiempos"]), 4)
            por_categoria_stats[cat] = {
                "promedio_seg": promedio,
                "total_seg": total,
            }

    report = {
        "total_pdfs": len(results),
        "tiempo_total_seg": tiempo_total_acumulado,
        "tiempo_promedio_seg": tiempo_promedio,
        "tiempo_mediano_seg": tiempo_mediano,
        "tiempo_min_seg": tiempo_min,
        "tiempo_max_seg": tiempo_max,
        "percentil_95_seg": tiempo_p95,
        "por_categoria": por_categoria_stats,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "timing_pymupdf.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Reporte de timing guardado en: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae abstracts de PDFs usando PyMuPDF."
    )
    parser.add_argument("--test", action="store_true", help="Procesa solo 3 PDFs para prueba")
    args = parser.parse_args()

    pdfs_dir = Path("data/pdfs")
    metadata_path = Path("data/raw/metadata.json")
    output_path = Path("data/interim/extracted_pymupdf.json")
    raw_dir = Path("data/interim/raw_pymupdf")

    if not pdfs_dir.exists():
        sys.exit(f"Error: {pdfs_dir} no existe")

    if not metadata_path.exists():
        sys.exit(f"Error: {metadata_path} no existe")

    # Leer metadata para obtener arxiv_id y category
    metadata = json.loads(metadata_path.read_text())

    if not metadata:
        sys.exit("Error: metadata.json está vacío")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Leer resultados existentes para reanudación
    processed_ids = set()
    if output_path.exists():
        existing_results = json.loads(output_path.read_text())
        processed_ids = {r["arxiv_id"] for r in existing_results}

    # Filtrar artículos no procesados
    metadata_to_process = [a for a in metadata if a["arxiv_id"] not in processed_ids]

    if not metadata_to_process:
        total_processed = len(processed_ids) if processed_ids else 0
        print(f"Todos los {total_processed} artículos ya están procesados.")
        sys.exit(0)

    # Imprimir estado de progreso
    if processed_ids:
        print(f"Progreso previo: {len(processed_ids)} artículos procesados")
        print(f"Pendientes: {len(metadata_to_process)} artículos\n")
    else:
        print(f"Iniciando desde cero: {len(metadata_to_process)} artículos\n")

    # Modo test: procesar solo 3 primeros
    if args.test:
        metadata_to_process = metadata_to_process[:3]
        print(f"Modo TEST: procesando solo {len(metadata_to_process)} artículos\n")

    success_count = 0
    failed_count = 0

    for i, article in enumerate(metadata_to_process, 1):
        arxiv_id = article.get("arxiv_id", "")
        category = article.get("primary_category", "unknown")
        article_title = article.get("title", "")

        pdf_path = pdfs_dir / category / f"{arxiv_id}.pdf"

        print(
            f"[{i}/{len(metadata_to_process)}] {arxiv_id} ({category})...",
            end=" ",
            flush=True,
        )

        if not pdf_path.exists():
            print("PDF no encontrado")
            continue

        result = process_pdf(pdf_path, raw_dir)
        result["arxiv_id"] = arxiv_id
        result["category"] = category
        # Usar título extraído del PDF si está disponible, sino del metadata
        if not result["title"]:
            result["title"] = article_title

        # Guardado incremental atómico
        append_to_json(result, output_path)

        if result["status"] == "success":
            success_count += 1
            print("OK")
        else:
            failed_count += 1
            print("FALLO")

    # Recargar todos los resultados para reporte final
    all_results = json.loads(output_path.read_text())

    # Generar reporte de timing con todos los resultados
    generate_timing_report(all_results, Path("reports"))

    print(f"\n{'='*60}")
    if args.test:
        print(f"Test completado: {success_count + failed_count} artículos procesados")
        print(f"Verificar {output_path} antes de continuar")
    else:
        print(f"Guardado: {output_path}")
        print(f"Nuevos extraídos: {success_count} | Nuevos fallos: {failed_count}")
        print(f"Total acumulado: {len(all_results)}")
    print(f"Raw guardado en: {raw_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

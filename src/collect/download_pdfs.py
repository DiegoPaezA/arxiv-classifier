import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DELAY_SECONDS = 3


def _save_report(
    downloaded: int, failed: int, skipped: int, failed_ids: list[str]
) -> None:
    """Guarda reporte de descargas en JSON."""
    report_path = Path("data/raw/download_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "total": downloaded + failed + skipped,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "failed_ids": failed_ids,
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Reporte guardado: {report_path}")


def download_pdf(url: str, output_path: Path) -> bool:
    """Descarga PDF de URL a output_path. Retorna True si exitoso."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
            output_path.write_bytes(data)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"    Error: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga PDFs de arXiv según metadata.json"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Modo prueba: descarga solo los primeros 3 PDFs.",
    )
    args = parser.parse_args()

    metadata_path = Path("data/raw/metadata.json")
    if not metadata_path.exists():
        sys.exit(f"Error: no se encontró {metadata_path}")

    articles = json.loads(metadata_path.read_text())

    if not articles:
        sys.exit("Error: metadata.json está vacío")

    if args.test:
        articles = articles[:3]
        print("[MODO TEST] Primeros 3 PDFs.\n")

    downloaded = 0
    failed = 0
    skipped = 0
    failed_ids = []

    for i, article in enumerate(articles, 1):
        arxiv_id = article.get("arxiv_id", "")
        pdf_url = article.get("pdf_url", "")
        category = article.get("primary_category", "unknown")
        title = (article.get("title", "")[:40] + "...") if article.get("title") else "sin título"

        if not pdf_url:
            print(f"[{i}/{len(articles)}] {arxiv_id} — SIN URL PDF")
            skipped += 1
            continue

        output_dir = Path("data/pdfs") / category
        output_path = output_dir / f"{arxiv_id}.pdf"

        if output_path.exists():
            print(f"[{i}/{len(articles)}] {arxiv_id} — YA EXISTE")
            skipped += 1
            continue

        print(f"[{i}/{len(articles)}] {arxiv_id} ({category})")
        print(f"  {title}")
        print(f"  >> Descargando desde {pdf_url[:60]}...")

        if download_pdf(pdf_url, output_path):
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  >> Guardado: {output_path}  ({size_mb:.2f} MB)")
            downloaded += 1
        else:
            failed += 1
            failed_ids.append(arxiv_id)
            print(f"  >> FALLO")

        if i < len(articles):
            time.sleep(DELAY_SECONDS)

    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"  Descargados: {downloaded}")
    print(f"  Fallos:      {failed}")
    print(f"  Omitidos:    {skipped}")
    print(f"{'='*60}")

    _save_report(downloaded, failed, skipped, failed_ids)


if __name__ == "__main__":
    main()

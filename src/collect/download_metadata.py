import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}
BATCH_SIZE = 200
DELAY_SECONDS = 3
MAX_RETRIES = 3


def _fetch(code: str, start: int, max_results: int) -> tuple[list[ET.Element], int]:
    params = urllib.parse.urlencode({
        "search_query": f"cat:{code}",
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    with urllib.request.urlopen(f"{ARXIV_API}?{params}", timeout=30) as resp:
        root = ET.fromstring(resp.read())
    total_el = root.find("opensearch:totalResults", NS)
    total = int(total_el.text) if total_el is not None else 0
    return root.findall("atom:entry", NS), total


def fetch_batch(code: str, start: int, max_results: int) -> tuple[list[ET.Element], int]:
    for attempt in range(MAX_RETRIES):
        try:
            return _fetch(code, start, max_results)
        except (urllib.error.URLError, urllib.error.HTTPError, ET.ParseError) as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = DELAY_SECONDS * (2 ** attempt)
            print(f"    [reintento {attempt + 1}/{MAX_RETRIES - 1}] {exc} — esperando {wait}s")
            time.sleep(wait)


def _strip_version(id_url: str) -> str:
    """'http://arxiv.org/abs/2301.00001v2' -> '2301.00001'"""
    raw = id_url.rstrip("/").split("/")[-1]
    base, sep, ver = raw.rpartition("v")
    return base if sep and ver.isdigit() else raw


def collect_category(
    code: str, n_per_class: int, total_offset: int = 0
) -> tuple[list[dict], dict[str, int]]:
    collected: list[dict] = []
    discarded = {"no_title": 0, "no_abstract": 0, "wrong_primary": 0}
    start = 0
    total_available: int | None = None
    batch_num = 0
    first_batch = True

    while len(collected) < n_per_class:
        if not first_batch:
            time.sleep(DELAY_SECONDS)
        first_batch = False
        batch_num += 1

        entries, total = fetch_batch(code, start, BATCH_SIZE)
        if total_available is None:
            total_available = total

        for entry in entries:
            if len(collected) >= n_per_class:
                break

            title_el = entry.find("atom:title", NS)
            summary_el = entry.find("atom:summary", NS)
            pc_el = entry.find("arxiv:primary_category", NS)

            title = (title_el.text or "").strip() if title_el is not None else ""
            abstract = (summary_el.text or "").strip() if summary_el is not None else ""
            primary = pc_el.get("term", "") if pc_el is not None else ""

            if not title:
                discarded["no_title"] += 1
                continue
            if not abstract:
                discarded["no_abstract"] += 1
                continue
            if primary != code:
                discarded["wrong_primary"] += 1
                continue

            id_el = entry.find("atom:id", NS)
            arxiv_id = _strip_version((id_el.text or "").strip()) if id_el is not None else ""

            all_categories = [
                c.get("term", "")
                for c in entry.findall("atom:category", NS)
                if c.get("term")
            ]

            published_el = entry.find("atom:published", NS)
            published = (published_el.text or "").strip() if published_el is not None else ""

            pdf_url = ""
            for link in entry.findall("atom:link", NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            collected.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract_api": abstract,
                "primary_category": primary,
                "all_categories": all_categories,
                "published": published,
                "pdf_url": pdf_url,
            })

        print(
            f"  Lote {batch_num}: {len(collected)}/{n_per_class} recolectados"
            f" | total acumulado: {total_offset + len(collected)}"
            f" | disponibles en arXiv: {total_available:,}"
        )

        start += len(entries)
        if not entries or start >= (total_available or 0):
            if len(collected) < n_per_class:
                print(f"  AVISO: arXiv agotado con solo {len(collected)}/{n_per_class} artículos primarios.")
            break

    return collected, discarded


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga metadata de arXiv por categoría.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Modo prueba: descarga solo 5 artículos de la primera categoría.",
    )
    args = parser.parse_args()

    config_path = Path("configs/categories.json")
    output_path = Path("data/raw/metadata.json")

    if not config_path.exists():
        sys.exit(f"Error: no se encontró {config_path}")

    categories = json.loads(config_path.read_text())

    if args.test:
        categories = [{**categories[0], "n_per_class": 5}]
        print("[MODO TEST] Primera categoría, 5 artículos.")

    required_fields = {"code", "name", "n_per_class"}
    for cat in categories:
        missing = required_fields - cat.keys()
        if missing:
            sys.exit(f"Error: campo(s) faltante(s) en {cat}: {missing}")

    if output_path.exists():
        resp = input(f"\n{output_path} ya existe. ¿Sobreescribir? [s/N]: ").strip().lower()
        if resp not in ("s", "si", "sí", "y", "yes"):
            print("Operación cancelada.")
            sys.exit(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_articles: list[dict] = []
    summary: list[dict] = []
    n_total = len(categories)

    for i, cat in enumerate(categories):
        code = cat["code"]
        name = cat["name"]
        n_per_class = int(cat["n_per_class"])

        print(f"\n[{i + 1}/{n_total}] {code} — {name}  (objetivo: {n_per_class})")

        if i > 0:
            time.sleep(DELAY_SECONDS)

        try:
            articles, discarded = collect_category(
                code, n_per_class, total_offset=len(all_articles)
            )
        except (urllib.error.URLError, urllib.error.HTTPError, ET.ParseError) as exc:
            print(f"  ERROR fatal: {exc}")
            articles = []
            discarded = {"no_title": 0, "no_abstract": 0, "wrong_primary": 0}

        all_articles.extend(articles)
        summary.append({
            "code": code,
            "name": name,
            "target": n_per_class,
            "collected": len(articles),
            "discarded": discarded,
        })

        total_disc = sum(discarded.values())
        print(
            f"  >> {len(articles)}/{n_per_class} recolectados"
            f" | descartados: {total_disc}"
            f"  (cat. incorrecta: {discarded['wrong_primary']},"
            f" sin título: {discarded['no_title']},"
            f" sin abstract: {discarded['no_abstract']})"
        )

    output_path.write_text(json.dumps(all_articles, indent=2, ensure_ascii=False))
    print(f"\nGuardado: {output_path}  ({len(all_articles)} artículos en total)")

    _print_summary(summary)


def _print_summary(summary: list[dict]) -> None:
    cw = [8, 36, 8, 8, 14, 9, 9]
    header = (
        f"{'Código':<{cw[0]}}  {'Nombre':<{cw[1]}}  {'Objetivo':>{cw[2]}}  "
        f"{'Recol.':>{cw[3]}}  {'Cat. incorrecta':>{cw[4]}}  "
        f"{'Sin ttl':>{cw[5]}}  {'Sin abs':>{cw[6]}}"
    )
    sep_thick = "=" * len(header)
    sep_thin = "-" * len(header)

    print(f"\n{sep_thick}")
    print("RESUMEN FINAL")
    print(sep_thick)
    print(header)
    print(sep_thin)

    for s in summary:
        d = s["discarded"]
        print(
            f"{s['code']:<{cw[0]}}  {s['name']:<{cw[1]}}  {s['target']:>{cw[2]}}  "
            f"{s['collected']:>{cw[3]}}  {d['wrong_primary']:>{cw[4]}}  "
            f"{d['no_title']:>{cw[5]}}  {d['no_abstract']:>{cw[6]}}"
        )

    print(sep_thin)
    total_col = sum(s["collected"] for s in summary)
    total_tgt = sum(s["target"] for s in summary)
    total_d = {
        k: sum(s["discarded"][k] for s in summary)
        for k in ("no_title", "no_abstract", "wrong_primary")
    }
    print(
        f"{'TOTAL':<{cw[0]}}  {'': <{cw[1]}}  {total_tgt:>{cw[2]}}  "
        f"{total_col:>{cw[3]}}  {total_d['wrong_primary']:>{cw[4]}}  "
        f"{total_d['no_title']:>{cw[5]}}  {total_d['no_abstract']:>{cw[6]}}"
    )
    print(sep_thick)


if __name__ == "__main__":
    main()

import json
import time
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

from arxiv_client import fetch_category_stats

MIN_ARTICLES = 5000
SAMPLE_SIZE = 100
DELAY_SECONDS = 3


def main():
    config_path = Path("configs/categories_candidates.json")
    categories = json.loads(config_path.read_text())

    col_w = [6, 36, 14, 12]
    header = (
        f"{'Code':<{col_w[0]}}  {'Name':<{col_w[1]}}  "
        f"{'Total':>{col_w[2]}}  {'Primary %':>{col_w[3]}}"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    results = []
    for i, cat in enumerate(categories):
        code, name = cat["code"], cat["name"]
        try:
            total, primary_pct = fetch_category_stats(code, max_results=SAMPLE_SIZE)
            status = ""
        except (urllib.error.URLError, urllib.error.HTTPError, ET.ParseError) as exc:
            total, primary_pct = 0, 0.0
            status = f"  [ERROR: {exc}]"

        results.append({"code": code, "name": name, "total": total, "primary_pct": primary_pct})
        print(
            f"{code:<{col_w[0]}}  {name:<{col_w[1]}}  "
            f"{total:>{col_w[2]},}  {primary_pct:>{col_w[3]}.1f}%{status}"
        )

        if i < len(categories) - 1:
            time.sleep(DELAY_SECONDS)

    print(separator)

    viable = [r for r in results if r["total"] >= MIN_ARTICLES]
    print(f"\nCategorias viables (total >= {MIN_ARTICLES:,}):")
    if viable:
        for r in viable:
            print(f"  {r['code']:<{col_w[0]}}  {r['name']}")
    else:
        print("  Ninguna")

    output_path = Path("data/results/categories_stats.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResultados guardados en {output_path}")


if __name__ == "__main__":
    main()

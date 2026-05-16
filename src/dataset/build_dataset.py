#!/usr/bin/env python3
"""Build unified dataset for arXiv paper classification.

Dataset conserves all three abstract sources (api, pymupdf, docling).
Source selection happens only during training, not here.
"""

import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    """Save data as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_label_map(categories: list[dict]) -> dict[str, int]:
    """Build label to label_id mapping, sorted alphabetically by code."""
    codes = sorted([cat["code"] for cat in categories])
    return {code: idx for idx, code in enumerate(codes)}


def process_articles(
    extracted_path: str,
    categories_path: str,
    test_mode: bool = False,
) -> tuple[list[dict], dict[str, int], dict]:
    """
    Process articles from extracted.json, conserving all abstract sources.

    Returns:
        (dataset, label_map, statistics)
    """
    articles = load_json(extracted_path)
    categories = load_json(categories_path)

    # Build label map
    label_map = build_label_map(categories)
    category_codes = {cat["code"] for cat in categories}

    # Track statistics
    stats = {
        "total_processed": 0,
        "total_accepted": 0,
        "discarded_empty_api": 0,
        "discarded_unknown_category": 0,
        "by_class": {},
        "abstract_sources": {
            "abstract_api": 0,
            "abstract_pymupdf": 0,
            "abstract_docling": 0,
        },
    }

    # Initialize class stats
    for code in label_map:
        stats["by_class"][code] = {
            "count": 0,
            "label_id": label_map[code],
        }

    dataset = []

    # Process only first 20 articles in test mode
    articles_to_process = articles[:20] if test_mode else articles

    for article in articles_to_process:
        stats["total_processed"] += 1

        # Check if abstract_api is present and non-empty (primary source)
        abstract_api = article.get("abstract_api", "").strip()
        if not abstract_api:
            stats["discarded_empty_api"] += 1
            continue

        # Check if category is in categories.json
        category = article.get("category", "").strip()
        if category not in category_codes:
            stats["discarded_unknown_category"] += 1
            continue

        # Extract all abstract sources
        abstract_pymupdf = article.get("abstract_pymupdf", "").strip()
        abstract_docling = article.get("abstract_docling", "").strip()

        # Get extraction status
        extraction_status = article.get("extraction_status", {})

        # If abstract is empty, mark extraction as failed
        if not abstract_pymupdf:
            extraction_status = dict(extraction_status)
            extraction_status["pymupdf"] = "failed"

        if not abstract_docling:
            extraction_status = dict(extraction_status)
            extraction_status["docling"] = "failed"

        # Count available abstracts by source
        if abstract_api:
            stats["abstract_sources"]["abstract_api"] += 1
        if abstract_pymupdf:
            stats["abstract_sources"]["abstract_pymupdf"] += 1
        if abstract_docling:
            stats["abstract_sources"]["abstract_docling"] += 1

        # Build record with all sources
        label_id = label_map[category]
        record = {
            "arxiv_id": article.get("arxiv_id"),
            "title": article.get("title"),
            "label": category,
            "label_id": label_id,
            "abstract_api": abstract_api,
            "abstract_pymupdf": abstract_pymupdf,
            "abstract_docling": abstract_docling,
            "score_pymupdf": article.get("score_pymupdf"),
            "score_docling": article.get("score_docling"),
            "extraction_status": extraction_status,
        }

        dataset.append(record)
        stats["total_accepted"] += 1
        stats["by_class"][category]["count"] += 1

    return dataset, label_map, stats


def print_summary(stats: dict, dataset: list[dict], total_articles: int) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("DATASET BUILD SUMMARY")
    print("=" * 80)

    print(f"\nTotal articles processed: {stats['total_processed']}")
    print(f"Total articles accepted: {stats['total_accepted']}")
    print(f"Discarded (empty abstract_api): {stats['discarded_empty_api']}")
    print(f"Discarded (unknown category): {stats['discarded_unknown_category']}")

    print("\n" + "-" * 80)
    print("ARTICLES PER CLASS")
    print("-" * 80)

    for code in sorted(stats["by_class"].keys()):
        class_stats = stats["by_class"][code]
        count = class_stats["count"]
        label_id = class_stats["label_id"]
        print(f"  {code} (label_id={label_id}): {count:4d} articles")

    print("\n" + "-" * 80)
    print("ABSTRACT SOURCES AVAILABILITY")
    print("-" * 80)

    for source, count in stats["abstract_sources"].items():
        if total_articles > 0:
            percentage = (count / total_articles) * 100
            print(f"  {source:20s}: {count:4d} ({percentage:5.1f}%)")
        else:
            print(f"  {source:20s}: {count:4d} (N/A)")

    print("\n" + "=" * 80)


def main() -> None:
    """Main entry point."""
    test_mode = "--test" in sys.argv

    # Paths
    extracted_path = "data/interim/extracted.json"
    categories_path = "configs/categories.json"
    label_map_path = "configs/label_map.json"
    dataset_path = "data/processed/dataset.json"

    print("Building unified dataset from extracted.json...\n")

    # Process articles
    dataset, label_map, stats = process_articles(
        extracted_path,
        categories_path,
        test_mode=test_mode,
    )

    if test_mode:
        # Test mode: print sample records and summary
        print("\n" + "=" * 80)
        print("TEST MODE - First 2 records:")
        print("=" * 80)

        for i, record in enumerate(dataset[:2], 1):
            print(f"\nRecord {i}:")
            print(json.dumps(record, indent=2))

        print_summary(stats, dataset, stats["total_processed"])
        print("\n(No files saved in test mode)\n")
    else:
        # Normal mode: save files
        save_json(label_map, label_map_path)
        save_json(dataset, dataset_path)

        print_summary(stats, dataset, stats["total_accepted"])
        print(f"\nFiles saved:")
        print(f"  - {label_map_path}")
        print(f"  - {dataset_path}\n")


if __name__ == "__main__":
    main()

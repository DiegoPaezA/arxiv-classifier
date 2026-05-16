#!/usr/bin/env python3
"""Split unified dataset into stratified train/val/test splits.

Preserves all record fields from the original dataset.json.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split


def load_json(path: str) -> any:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: any, path: str) -> None:
    """Save data as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_class_distribution(dataset: list[dict]) -> dict[str, int]:
    """Count articles per class."""
    distribution = defaultdict(int)
    for record in dataset:
        label = record.get("label")
        distribution[label] += 1
    return dict(sorted(distribution.items()))


def split_dataset(
    dataset: list[dict],
    test_size: float = 0.3,
    val_ratio: float = 0.5,
    random_state: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split dataset into train/val/test with stratification by label.

    Proportions: 70% train, 15% val, 15% test

    Args:
        dataset: List of article records
        test_size: Proportion for test+val (0.3 = 70% train, 30% test+val)
        val_ratio: Of the test_size, how much goes to val (0.5 = 15% val, 15% test)
        random_state: Seed for reproducibility (42)

    Returns:
        (train, val, test)
    """
    labels = [record["label"] for record in dataset]

    # First split: 70% train, 30% temp (val+test)
    train, temp = train_test_split(
        dataset,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    # Second split: split temp into val and test (50/50 of the 30%)
    temp_labels = [record["label"] for record in temp]
    val, test = train_test_split(
        temp,
        test_size=val_ratio,
        stratify=temp_labels,
        random_state=random_state,
    )

    return train, val, test


def print_distribution_table(dataset: list[dict], splits: dict[str, list[dict]]) -> None:
    """Print distribution table for all splits."""
    all_labels = sorted(set(r["label"] for r in dataset))

    # Calculate distributions
    distributions = {}
    for split_name, split_data in splits.items():
        distributions[split_name] = get_class_distribution(split_data)

    total_dist = get_class_distribution(dataset)

    # Print header
    print("\n" + "=" * 70)
    print("DATASET SPLIT DISTRIBUTION")
    print("=" * 70)
    print(f"{'Label':<10} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 70)

    # Print each class
    for label in all_labels:
        train_count = distributions["train"].get(label, 0)
        val_count = distributions["val"].get(label, 0)
        test_count = distributions["test"].get(label, 0)
        total_count = total_dist.get(label, 0)

        print(f"{label:<10} {train_count:>8} {val_count:>8} {test_count:>8} {total_count:>8}")

        # Check for empty splits
        if train_count == 0 or val_count == 0 or test_count == 0:
            print(f"  ⚠️  WARNING: class {label} has empty split!")

    # Print totals
    print("-" * 70)
    train_total = sum(distributions["train"].values())
    val_total = sum(distributions["val"].values())
    test_total = sum(distributions["test"].values())
    overall_total = len(dataset)

    print(f"{'TOTAL':<10} {train_total:>8} {val_total:>8} {test_total:>8} {overall_total:>8}")
    print("=" * 70 + "\n")


def main() -> None:
    """Main entry point."""
    test_mode = "--test" in sys.argv

    # Paths
    dataset_path = "data/processed/dataset.json"
    train_path = "data/processed/train.json"
    val_path = "data/processed/val.json"
    test_path = "data/processed/test.json"

    # Load unified dataset
    print(f"Loading {dataset_path}...")
    dataset = load_json(dataset_path)
    print(f"Loaded {len(dataset)} articles\n")

    # Split dataset (70/15/15 stratified by label)
    print("Splitting with stratification by label (seed=42)...")
    train, val, test = split_dataset(
        dataset,
        test_size=0.3,
        val_ratio=0.5,
        random_state=42,
    )

    # Create splits dict for printing
    splits = {
        "train": train,
        "val": val,
        "test": test,
    }

    # Print distribution
    print_distribution_table(dataset, splits)

    if test_mode:
        print("(Test mode: no files saved)\n")
    else:
        # Save splits
        print("Saving splits...")
        save_json(train, train_path)
        save_json(val, val_path)
        save_json(test, test_path)
        print(f"✓ {train_path}")
        print(f"✓ {val_path}")
        print(f"✓ {test_path}\n")


if __name__ == "__main__":
    main()

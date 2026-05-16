#!/usr/bin/env python3
"""
Evaluate fine-tuned SciBERT models on test set.

Supports evaluation across different abstract sources (api, pymupdf, docling)
and generates comprehensive metrics including per-class performance and
confusion matrix.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_json(path: str):
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    """Save data as JSON with pretty formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def find_best_checkpoint(base_model_path: str, source: str) -> tuple[str, str]:
    """
    Find the best checkpoint for a trained model.

    Strategy:
    1. Read reports/best_checkpoint_{source}.json (written by train_bert.py)
    2. Fallback: use checkpoint with highest number

    Args:
        base_model_path: Path like "models/scibert_api"
        source: Abstract source ("api", "pymupdf", "docling")

    Returns:
        (checkpoint_path, selection_method) tuple
    """
    import glob as _glob

    # Strategy 1: Read best checkpoint info saved by train_bert.py
    best_checkpoint_path = f"reports/best_checkpoint_{source}.json"
    if os.path.exists(best_checkpoint_path):
        with open(best_checkpoint_path) as f:
            info = json.load(f)
        checkpoint = info["best_checkpoint"]
        return checkpoint, "mejor val_f1_macro (best_checkpoint.json)"

    # Fallback: use checkpoint with highest number
    checkpoints = _glob.glob(f"{base_model_path}/checkpoint-*")
    if checkpoints:
        checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        return checkpoint, "checkpoint más reciente (fallback)"

    return base_model_path, "sin checkpoints válidos"


def build_inverse_label_map(label_map: dict) -> dict:
    """Convert label_map {label: id} to inverse {id: label}."""
    return {v: k for k, v in label_map.items()}


def load_and_filter_test_data(source: str = "api", test_mode: bool = False) -> tuple[list, int, int]:
    """
    Load test.json and filter articles with valid abstracts for the source.

    Returns:
        (filtered_articles, num_used, num_discarded)
    """
    test_data = load_json("data/processed/test.json")

    # In test mode, use only first 30 articles
    if test_mode:
        test_data = test_data[:30]

    original_count = len(test_data)
    abstract_field = f"abstract_{source}"

    # Filter: keep only articles with non-empty abstract for the selected source
    filtered = [
        article for article in test_data
        if article.get(abstract_field, "").strip() != ""
    ]

    discarded = original_count - len(filtered)

    return filtered, len(filtered), discarded


def tokenize_abstracts(
    articles: list,
    tokenizer,
    source: str = "api",
    max_length: int = 512,
):
    """
    Tokenize abstracts from articles.

    Uses the specified source field (abstract_api, abstract_pymupdf, abstract_docling)
    and applies consistent tokenization settings:
    - max_length=512: truncate long abstracts
    - truncation=True: enable truncation
    - padding="max_length": pad all sequences to max_length
    """
    abstract_field = f"abstract_{source}"
    abstracts = [article.get(abstract_field, "").strip() for article in articles]

    # Tokenize all abstracts
    encodings = tokenizer(
        abstracts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    return encodings


def run_inference(
    model,
    encodings,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Run model inference on tokenized abstracts in batches.

    Returns logits for each sample and shows progress bar.
    """
    model.to(device)
    model.eval()

    all_logits = []
    num_samples = encodings["input_ids"].shape[0]

    # Process in batches with progress bar
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Inferring batches"):
            batch_end = min(i + batch_size, num_samples)

            # Extract batch
            batch_input_ids = encodings["input_ids"][i:batch_end].to(device)
            batch_attention_mask = encodings["attention_mask"][i:batch_end].to(device)

            # Forward pass
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )

            # Collect logits
            all_logits.append(outputs.logits.cpu().detach().numpy())

    # Concatenate all batches
    all_logits = np.vstack(all_logits)

    return all_logits


def logits_to_predictions(logits):
    """Convert logits to predictions and probabilities using softmax."""
    # Softmax to get probabilities
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    # Argmax to get predicted class
    predictions = np.argmax(logits, axis=1)

    return predictions, probabilities


def calculate_metrics(y_true, y_pred, y_proba, label_map: dict):
    """
    Calculate global and per-class metrics.

    Returns:
        (global_metrics, per_class_metrics, confusion_mat)
    """
    num_classes = len(label_map)
    inverse_map = build_inverse_label_map(label_map)

    # Global metrics
    global_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-class metrics
    per_class_metrics = {}

    # Get precision, recall, f1 per class
    # Use labels=range(num_classes) to ensure all classes are represented
    # even if some don't appear in the data
    precisions = precision_score(
        y_true, y_pred, average=None, zero_division=0, labels=range(num_classes)
    )
    recalls = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=range(num_classes)
    )
    f1_scores_per_class = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=range(num_classes)
    )

    # Count support (number of samples per class)
    for class_id in range(num_classes):
        class_label = inverse_map[class_id]
        support = np.sum(y_true == class_id)

        per_class_metrics[class_label] = {
            "precision": float(precisions[class_id]),
            "recall": float(recalls[class_id]),
            "f1": float(f1_scores_per_class[class_id]),
            "support": int(support),
        }

    # Confusion matrix
    confusion_mat = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    confusion_mat = confusion_mat.tolist()

    return global_metrics, per_class_metrics, confusion_mat


def build_predictions_list(
    articles: list,
    y_true,
    y_pred,
    y_proba,
    label_map: dict,
):
    """
    Build list of detailed predictions for each article.

    Each entry contains: arxiv_id, true_label, predicted_label, confidence, correct, probabilities
    """
    inverse_map = build_inverse_label_map(label_map)
    predictions_list = []

    for i, article in enumerate(articles):
        true_id = int(y_true[i])
        pred_id = int(y_pred[i])

        true_label = inverse_map[true_id]
        pred_label = inverse_map[pred_id]

        # Confidence is the max probability
        confidence = float(y_proba[i, pred_id])

        # Correctness
        correct = (true_id == pred_id)

        # All probabilities per class
        probs_dict = {
            inverse_map[class_id]: float(y_proba[i, class_id])
            for class_id in range(len(label_map))
        }

        predictions_list.append({
            "arxiv_id": article["arxiv_id"],
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": confidence,
            "correct": correct,
            "probabilities": probs_dict,
        })

    return predictions_list


def find_worst_f1_classes(per_class_metrics: dict, top_n: int = 3):
    """Find classes with lowest F1 scores."""
    f1_tuples = [
        (label, metrics["f1"])
        for label, metrics in per_class_metrics.items()
    ]

    # Sort by F1 ascending
    f1_tuples.sort(key=lambda x: x[1])

    return f1_tuples[:top_n]


def print_evaluation_summary(
    source: str,
    model_path: str,
    num_articles: int,
    num_discarded: int,
    global_metrics: dict,
    per_class_metrics: dict,
):
    """Print formatted evaluation summary to console."""
    print("\n" + "═" * 50)
    print("EVALUACIÓN — SciBERT fine-tuning")
    print("═" * 50)
    print(f"Fuente      : abstract_{source}")
    print(f"Modelo      : {model_path}")
    print(f"Artículos   : {num_articles} ({num_discarded} descartados)")
    print("═" * 50)

    # Global metrics
    print("\nMÉTRICAS GLOBALES")
    print(f"Accuracy        : {global_metrics['accuracy']:.4f}")
    print(f"F1 macro        : {global_metrics['f1_macro']:.4f}")
    print(f"F1 weighted     : {global_metrics['f1_weighted']:.4f}")
    print(f"Precision macro : {global_metrics['precision_macro']:.4f}")
    print(f"Recall macro    : {global_metrics['recall_macro']:.4f}")

    # Per-class metrics
    print("\nMÉTRICAS POR CLASE")
    print(f"{'Clase':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print("-" * 63)

    for label, metrics in per_class_metrics.items():
        print(
            f"{label:<15} {metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['support']:<12}"
        )

    # Worst performing classes
    worst = find_worst_f1_classes(per_class_metrics, top_n=3)
    print("\nCLASES CON MENOR F1 (top 3 peores):")
    for rank, (label, f1) in enumerate(worst, 1):
        print(f"  {rank}. {label:<20} → F1: {f1:.4f}")

    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned SciBERT on test set"
    )
    parser.add_argument(
        "--source",
        choices=["api", "pymupdf", "docling"],
        default="api",
        help="Abstract source to use (default: api)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run on first 30 test articles only",
    )
    args = parser.parse_args()

    source = args.source
    test_mode = args.test

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load label mapping
    print("Loading label map...")
    label_map = load_json("configs/label_map.json")
    inverse_label_map = build_inverse_label_map(label_map)
    label_order = [inverse_label_map[i] for i in range(len(label_map))]

    # Load and filter test data
    print(f"Loading test data (source: {source})...")
    articles, num_used, num_discarded = load_and_filter_test_data(source, test_mode)
    print(f"Using {num_used} articles ({num_discarded} discarded)")

    # Extract ground truth labels
    y_true = np.array([label_map[article["label"]] for article in articles])

    # Find and load best checkpoint
    base_model_path = f"models/scibert_{source}"
    print(f"\nBuscando mejor checkpoint:")
    print(f"  Modelo base  : {base_model_path}")

    model_path, selection_method = find_best_checkpoint(base_model_path, source)

    if model_path != base_model_path:
        print(f"  Checkpoint   : {model_path}")
        print(f"  Seleccionado : {selection_method}")
    else:
        print(f"  Seleccionado : {selection_method}")

    print(f"\nLoading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Tokenize abstracts
    print("Tokenizing abstracts...")
    encodings = tokenize_abstracts(articles, tokenizer, source)

    # Run inference
    print("Running inference...")
    logits = run_inference(model, encodings, device=device)

    # Convert to predictions and probabilities
    y_pred, y_proba = logits_to_predictions(logits)

    # Calculate metrics
    print("Calculating metrics...")
    global_metrics, per_class_metrics, confusion_mat = calculate_metrics(
        y_true, y_pred, y_proba, label_map
    )

    # Build detailed predictions list
    predictions_list = build_predictions_list(
        articles, y_true, y_pred, y_proba, label_map
    )

    # Build output JSON
    output_data = {
        "source": source,
        "model_path": model_path,
        "test_articles": num_used,
        "discarded": num_discarded,
        "metrics": global_metrics,
        "per_class": per_class_metrics,
        "confusion_matrix": confusion_mat,
        "label_order": label_order,
        "predictions": predictions_list,
    }

    # Save results
    suffix = "_test" if test_mode else ""
    output_path = f"reports/evaluation_results_{source}{suffix}.json"
    print(f"Saving results to {output_path}...")
    save_json(output_data, output_path)

    # Print summary
    print_evaluation_summary(
        source,
        model_path,
        num_used,
        num_discarded,
        global_metrics,
        per_class_metrics,
    )

    print(f"✓ Guardado en: {output_path}\n")


if __name__ == "__main__":
    main()

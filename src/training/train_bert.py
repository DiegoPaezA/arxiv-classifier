#!/usr/bin/env python3
"""
Fine-tune SciBERT for arXiv paper classification.

This script trains allenai/scibert_scivocab_uncased on a multi-class
classification task across computer science categories using a manual
PyTorch training loop instead of HuggingFace Trainer.

Main features:
- Supports different abstract sources: api, pymupdf, docling.
- Supports a lightweight test mode.
- Uses dynamic padding per batch to reduce memory usage.
- Uses gradient accumulation for a larger effective batch size.
- Saves the best model according to validation macro F1.
- Saves training history and checkpoint metadata.
"""

# Load environment variables before importing transformers
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


VALID_SOURCES = ["api", "pymupdf", "docling"]
MODEL_NAME = "allenai/scibert_scivocab_uncased"
MAX_LENGTH = 512
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
ACCUMULATION_STEPS = 2
DEFAULT_NUM_EPOCHS = 3
TEST_NUM_EPOCHS = 1
SEED = 42


class AbstractDataset(Dataset):
    """
    Dataset that tokenizes abstracts on demand.

    Dynamic padding is handled by DataCollatorWithPadding in the DataLoader.
    This avoids padding the full dataset to 512 tokens in advance.

    Note: __getitem__ returns a plain dict (not BatchEncoding) to ensure
    DataCollatorWithPadding handles the 'labels' field correctly across
    all versions of transformers.
    """

    def __init__(self, data: list[dict], tokenizer, label_map: dict, source: str = "api"):
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.abstract_field = f"abstract_{source}"
        self.texts = [record.get(self.abstract_field, "").strip() for record in data]
        self.labels = [label_map[record["label"]] for record in data]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        # Tokenize on demand — padding handled by DataCollatorWithPadding
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            truncation=True,
        )
        # Return plain dict to avoid BatchEncoding compatibility issues
        # with DataCollatorWithPadding across different transformers versions
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": self.labels[idx],
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune SciBERT for arXiv paper classification."
    )
    parser.add_argument(
        "--source",
        default="api",
        choices=VALID_SOURCES,
        help="Abstract source to use: api, pymupdf, or docling.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode using fewer examples and one epoch.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_NUM_EPOCHS}). "
             "Ignored in test mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def detect_device_and_batch_size() -> tuple[torch.device, int]:
    """
    Detect available device and set a reasonable batch size.

    Batch size choices:
    - >= 8GB GPU: batch_size=32 (fits comfortably)
    - >= 6GB GPU: batch_size=16 (safe for SciBERT + 512 tokens)
    - CPU or < 6GB: batch_size=8 (conservative)

    Effective batch size = batch_size * ACCUMULATION_STEPS.

    Returns:
        tuple with the selected device and physical batch size.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print("\n" + "=" * 70)
        print("DEVICE CONFIGURATION")
        print("=" * 70)
        print(f"GPU detected: {gpu_name}")
        print(f"Total GPU memory: {gpu_memory_gb:.2f} GB")

        if gpu_memory_gb >= 8:
            batch_size = 32
            print(f"Memory >= 8GB -> batch_size = {batch_size}")
        elif gpu_memory_gb >= 6:
            batch_size = 16
            print(f"Memory >= 6GB -> batch_size = {batch_size}")
        else:
            batch_size = 8
            print(f"Memory < 6GB -> batch_size = {batch_size}")
    else:
        device = torch.device("cpu")
        batch_size = 8
        print("\n" + "=" * 70)
        print("DEVICE CONFIGURATION")
        print("=" * 70)
        print("No GPU detected, using CPU")
        print(f"CPU mode -> batch_size = {batch_size}")

    print(f"Effective batch size with accumulation: {batch_size * ACCUMULATION_STEPS}")
    print("=" * 70 + "\n")

    return device, batch_size


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    """Save data as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_splits(
    source: str = "api",
    test_mode: bool = False,
) -> tuple[list, list, list, dict, dict]:
    """
    Load train, validation, and test splits from the unified dataset.

    The dataset contains abstracts from all extraction sources (api, pymupdf,
    docling). Source selection is handled by filtering on the corresponding
    abstract field. Articles with empty abstracts for the selected source
    are discarded to keep the experiment clean.

    Args:
        source: Abstract source: api, pymupdf, or docling.
        test_mode: If True, use fewer examples for a fast sanity check.

    Returns:
        train, val, test, label_map, discarded_counts.
    """
    print("Loading unified dataset splits...")

    train = load_json("data/processed/train.json")
    val = load_json("data/processed/val.json")
    test = load_json("data/processed/test.json")
    label_map = load_json("configs/label_map.json")

    if test_mode:
        print("TEST MODE: using first 50 train and 20 validation articles")
        train = train[:50]
        val = val[:20]

    original_sizes = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }

    abstract_field = f"abstract_{source}"

    def has_valid_abstract(article: dict) -> bool:
        return article.get(abstract_field, "").strip() != ""

    train = [r for r in train if has_valid_abstract(r)]
    val = [r for r in val if has_valid_abstract(r)]
    test = [r for r in test if has_valid_abstract(r)]

    discarded_counts = {
        "train": original_sizes["train"] - len(train),
        "val": original_sizes["val"] - len(val),
        "test": original_sizes["test"] - len(test),
    }

    print(f"Train: {len(train)} articles")
    print(f"Val:   {len(val)} articles")
    print(f"Test:  {len(test)} articles")
    print(f"Labels: {len(label_map)} classes\n")

    return train, val, test, label_map, discarded_counts


def validate_data_splits(
    train: list,
    val: list,
    label_map: dict,
    source: str,
) -> None:
    """Validate that required data is available after filtering."""
    if len(train) == 0:
        raise ValueError(f"No training examples found for source='{source}'.")
    if len(val) == 0:
        raise ValueError(f"No validation examples found for source='{source}'.")
    if len(label_map) == 0:
        raise ValueError("Label map is empty.")


def build_dataloader(
    data: list[dict],
    tokenizer,
    label_map: dict,
    source: str = "api",
    batch_size: int = 32,
    shuffle: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Build a PyTorch DataLoader using dynamic padding.

    Tokenization is performed on demand in AbstractDataset.__getitem__.
    Padding is applied only within each batch by DataCollatorWithPadding,
    which is more memory-efficient than padding everything to MAX_LENGTH.

    Args:
        data: List of article dicts.
        tokenizer: HuggingFace tokenizer.
        label_map: Mapping from label name to integer id.
        source: Abstract source field to use.
        batch_size: Number of examples per batch.
        shuffle: True for training, False for validation/test.
        pin_memory: True when using GPU (speeds up CPU->GPU transfers).
    """
    dataset = AbstractDataset(
        data=data,
        tokenizer=tokenizer,
        label_map=label_map,
        source=source,
    )

    # DataCollatorWithPadding pads input_ids and attention_mask to the
    # longest sequence in each batch. Labels are passed through unchanged.
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        pin_memory=pin_memory,
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a batch to the selected device."""
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """
    Evaluate model on a dataloader.

    Runs inference in no_grad mode for efficiency.
    Computes loss, accuracy, macro F1, and weighted F1.

    Returns:
        val_loss, val_accuracy, val_f1_macro, val_f1_weighted.
    """
    if len(dataloader) == 0:
        raise ValueError("Cannot evaluate with an empty dataloader.")

    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)

            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
            labels = batch["labels"].detach().cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    # macro F1: equal weight to all classes regardless of frequency
    # weighted F1: weight by class support (accounts for imbalance)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, accuracy, f1_macro, f1_weighted


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_labels: int,
    device: torch.device,
    tokenizer,
    label_map: dict,
    source: str = "api",
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    test_mode: bool = False,
) -> tuple[list[dict], float]:
    """
    Manual PyTorch training loop for SciBERT fine-tuning.

    Uses the HuggingFace Trainer to avoid a bug in transformers 5.8.1
    where classifier weights are not updated when using Trainer directly.

    Key hyperparameters:
    - learning_rate=2e-5: standard for BERT fine-tuning (range: 1e-5 to 5e-5)
    - warmup_steps=10% of total: gradual LR increase prevents early instability
    - weight_decay=0.01: L2 regularization to prevent overfitting
    - accumulation_steps=2: effective batch doubles without extra GPU memory
    - clip_grad_norm=1.0: prevents exploding gradients common in transformers

    Args:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        num_labels: Number of output classes.
        device: Selected torch device.
        tokenizer: Tokenizer to save alongside the model.
        label_map: Mapping from label name to integer id.
        source: Abstract source, used to name output paths.
        num_epochs: Number of training epochs.
        test_mode: If True, skip model saving.

    Returns:
        history (list of per-epoch metrics) and best validation macro F1.
    """
    if len(train_loader) == 0:
        raise ValueError("Cannot train with an empty train_loader.")

    print(f"Loading model: {MODEL_NAME}...")

    # Store label mappings in model config so the saved model is self-contained
    label2id = label_map
    id2label = {idx: label for label, idx in label_map.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    # Snapshot initial classifier weights to verify they change after training
    initial_classifier_weight = model.classifier.weight.detach().cpu().clone()
    initial_classifier_bias = model.classifier.bias.detach().cpu().clone()

    # total optimizer steps account for gradient accumulation
    num_update_steps_per_epoch = math.ceil(len(train_loader) / ACCUMULATION_STEPS)
    total_steps = num_update_steps_per_epoch * num_epochs
    warmup_steps = int(0.1 * total_steps)

    # AdamW decouples weight decay from the gradient update (unlike Adam)
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Linear warmup then linear decay to 0 — standard for BERT fine-tuning
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1 = -float("inf")
    best_dir = f"models/scibert_{source}/best_model"
    history: list[dict] = []

    print("\nTraining configuration")
    print("=" * 70)
    print(f"Epochs                  : {num_epochs}")
    print(f"Physical batch size     : {train_loader.batch_size}")
    print(f"Accumulation steps      : {ACCUMULATION_STEPS}")
    print(f"Effective batch size    : {train_loader.batch_size * ACCUMULATION_STEPS}")
    print(f"Optimizer steps / epoch : {num_update_steps_per_epoch}")
    print(f"Total optimizer steps   : {total_steps}")
    print(f"Warmup steps            : {warmup_steps}")
    print(f"Learning rate           : {LEARNING_RATE}")
    print(f"Weight decay            : {WEIGHT_DECAY}")
    print("=" * 70 + "\n")

    print(f"Starting training ({num_epochs} epoch(s))...\n")

    for epoch in range(1, num_epochs + 1):

        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)  # set_to_none frees memory faster

        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)

            outputs = model(**batch)

            # Divide loss by accumulation steps so the gradient magnitude
            # stays consistent regardless of how many steps we accumulate
            loss = outputs.loss / ACCUMULATION_STEPS
            loss.backward()

            total_train_loss += outputs.loss.item()

            is_accumulation_step = (step + 1) % ACCUMULATION_STEPS == 0
            is_last_step = (step + 1) == len(train_loader)

            if is_accumulation_step or is_last_step:
                # Clip gradients before stepping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        avg_train_loss = total_train_loss / len(train_loader)

        # ── Validation phase ──────────────────────────────────────────────────
        val_loss, val_accuracy, val_f1_macro, val_f1_weighted = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"val_f1_macro={val_f1_macro:.4f} | "
            f"val_f1_weighted={val_f1_weighted:.4f}"
        )

        # Save model whenever validation F1 improves
        if not test_mode and val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  -> New best model saved (val_f1_macro={best_f1:.4f})")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1_macro": val_f1_macro,
            "val_f1_weighted": val_f1_weighted,
        })

    # ── Classifier update verification ────────────────────────────────────────
    # Checks that the classifier head actually learned during training.
    # Uses L2 norm of the delta instead of std, which is more sensitive
    # to small but real changes across all 10 x 768 = 7,680 parameters.
    final_classifier_weight = model.classifier.weight.detach().cpu()
    final_classifier_bias = model.classifier.bias.detach().cpu()

    weight_delta = torch.norm(
        final_classifier_weight - initial_classifier_weight
    ).item()
    bias_delta = torch.norm(
        final_classifier_bias - initial_classifier_bias
    ).item()

    print("\nClassifier update check")
    print("=" * 70)
    print(f"Classifier weight L2 delta : {weight_delta:.8f}")
    print(f"Classifier bias L2 delta   : {bias_delta:.8f}")

    # In test mode with few examples, even a small delta is acceptable
    threshold = 1e-6
    if weight_delta < threshold and bias_delta < threshold:
        print("WARNING: classifier parameters did not change. Review the training loop.")
    else:
        print("OK: classifier parameters updated correctly.")
    print("=" * 70 + "\n")

    if best_f1 == -float("inf"):
        best_f1 = max((e["val_f1_macro"] for e in history), default=0.0)

    return history, best_f1


def print_experiment_information(
    source: str,
    source_names: dict,
    train_data: list,
    val_data: list,
    test_data: list,
    discarded_counts: dict,
    num_epochs: int,
) -> None:
    """Print a compact summary of the experiment setup."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: SciBERT fine-tuning")
    print("=" * 70)
    print(f"Source : {source_names[source]}")
    print(f"Epochs : {num_epochs}")
    print()
    print(f"{'Split':<10} {'Original':>10} {'Discarded':>12} {'Used':>10}")
    print("-" * 50)
    print(f"{'train':<10} {discarded_counts['train'] + len(train_data):>10} "
          f"{discarded_counts['train']:>12} {len(train_data):>10}")
    print(f"{'val':<10} {discarded_counts['val'] + len(val_data):>10} "
          f"{discarded_counts['val']:>12} {len(val_data):>10}")
    print(f"{'test':<10} {discarded_counts['test'] + len(test_data):>10} "
          f"{discarded_counts['test']:>12} {len(test_data):>10}")
    print("=" * 70 + "\n")


def save_training_outputs(
    source: str,
    best_f1: float,
    history: list[dict],
) -> None:
    """Save best checkpoint metadata and training history to reports/."""
    best_dir = f"models/scibert_{source}/best_model"

    best_checkpoint_info = {
        "source": source,
        "best_checkpoint": best_dir,
        "best_metric": "f1_macro",
        "best_value": best_f1,
    }

    save_json(best_checkpoint_info, f"reports/best_checkpoint_{source}.json")
    save_json(history, f"reports/training_history_{source}.json")

    print(f"Model saved in           : {best_dir}")
    print(f"Checkpoint info saved to : reports/best_checkpoint_{source}.json")
    print(f"Training history saved to: reports/training_history_{source}.json")


def print_training_summary(history: list[dict]) -> None:
    """Print epoch-by-epoch metrics after training completes."""
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for e in history:
        print(f"\nEpoch {int(e['epoch'])}:")
        print(f"  Train loss      : {e['train_loss']:.4f}")
        print(f"  Val loss        : {e['val_loss']:.4f}")
        print(f"  Val accuracy    : {e['val_accuracy']:.4f}")
        print(f"  Val F1 macro    : {e['val_f1_macro']:.4f}")
        print(f"  Val F1 weighted : {e['val_f1_weighted']:.4f}")
    print("=" * 70 + "\n")


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()
    set_seed(args.seed)

    source = args.source
    test_mode = args.test
    num_epochs = TEST_NUM_EPOCHS if test_mode else args.epochs

    source_names = {
        "api":     "abstract_api",
        "pymupdf": "abstract_pymupdf",
        "docling": "abstract_docling",
    }

    device, batch_size = detect_device_and_batch_size()

    train_data, val_data, test_data, label_map, discarded_counts = load_splits(
        source=source,
        test_mode=test_mode,
    )

    validate_data_splits(
        train=train_data,
        val=val_data,
        label_map=label_map,
        source=source,
    )

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Building DataLoaders with dynamic padding...")
    pin_memory = device.type == "cuda"

    train_loader = build_dataloader(
        data=train_data,
        tokenizer=tokenizer,
        label_map=label_map,
        source=source,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )

    val_loader = build_dataloader(
        data=val_data,
        tokenizer=tokenizer,
        label_map=label_map,
        source=source,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    print_experiment_information(
        source=source,
        source_names=source_names,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        discarded_counts=discarded_counts,
        num_epochs=num_epochs,
    )

    history, best_f1 = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_labels=len(label_map),
        device=device,
        tokenizer=tokenizer,
        label_map=label_map,
        source=source,
        num_epochs=num_epochs,
        test_mode=test_mode,
    )

    if test_mode:
        print("Test mode completed successfully.")
        print("Model not saved in test mode.\n")
        return

    print()
    save_training_outputs(source=source, best_f1=best_f1, history=history)
    print_training_summary(history)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(130)
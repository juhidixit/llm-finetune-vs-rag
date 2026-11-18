"""Banking77 loading, splits and label verbalisation.

Banking77 (Casanueva et al., 2020, CC-BY-4.0) has 10,003 train and 3,080 test customer
messages over 77 fine-grained intents of a digital bank. We carve a stratified 10%
validation split out of train for model selection; the official test split is only
used for the final numbers in results/.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data/task-specific-datasets/banking_data")
SEED = 13


@dataclass(frozen=True)
class Example:
    text: str
    label: str  # verbalised label, e.g. "card arrival"


def verbalise(raw: str) -> str:
    """'Refund_not_showing_up' -> 'refund not showing up'."""
    return raw.replace("_", " ").strip().lower()


def _read(path: Path) -> list[Example]:
    with path.open(newline="", encoding="utf-8") as f:
        return [Example(r["text"].strip(), verbalise(r["category"])) for r in csv.DictReader(f)]


def load_banking77(data_dir: Path = DATA_DIR) -> tuple[list[Example], list[Example]]:
    """Return (train, test) exactly as published."""
    return _read(data_dir / "train.csv"), _read(data_dir / "test.csv")


def labels_of(examples: list[Example]) -> list[str]:
    return sorted({e.label for e in examples})


def stratified_split(examples: list[Example], frac: float = 0.1, seed: int = SEED) -> tuple[list[Example], list[Example]]:
    """Split into (rest, held) with `frac` of every label in `held`."""
    by = defaultdict(list)
    for e in examples:
        by[e.label].append(e)
    rng = random.Random(seed)
    rest, held = [], []
    for lab in sorted(by):
        items = by[lab][:]
        rng.shuffle(items)
        k = max(1, round(len(items) * frac))
        held += items[:k]
        rest += items[k:]
    return rest, held


def per_label_subset(examples: list[Example], n: int, seed: int = SEED) -> list[Example]:
    """At most `n` examples per label (for data-efficiency curves)."""
    by = defaultdict(list)
    for e in examples:
        by[e.label].append(e)
    rng = random.Random(seed)
    out = []
    for lab in sorted(by):
        items = by[lab][:]
        rng.shuffle(items)
        out += items[:n]
    rng.shuffle(out)
    return out


def holdout_intents(labels: list[str], k: int = 10, seed: int = SEED) -> list[str]:
    """Intents treated as 'launched after training' in the new-intent experiment."""
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(labels), k))


def splits(data_dir: Path = DATA_DIR) -> dict[str, list[Example]]:
    train_full, test = load_banking77(data_dir)
    train, val = stratified_split(train_full)
    return {"train": train, "val": val, "test": test, "train_full": train_full}

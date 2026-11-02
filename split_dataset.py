"""
Stratified 70/15/15 split of dataset/fetched_dataset into
dataset/train, dataset/validation and dataset/test.

Each class keeps the same proportion in every subset (here 50/50), so with
10,400 images per class the result is:
  train       14,560 = 7,280 single_person + 7,280 multiple_people
  validation   3,120 = 1,560 single_person + 1,560 multiple_people
  test         3,120 = 1,560 single_person + 1,560 multiple_people

Images are COPIED, so dataset/fetched_dataset is left untouched.
Run from the project root:  python split_dataset.py
"""
from collections import Counter
from pathlib import Path
import shutil
import sys

from sklearn.model_selection import train_test_split

DATASET = Path("dataset")
SOURCE = DATASET / "fetched_dataset"
CATEGORIES = ["single_person", "multiple_people"]
SPLITS = ["train", "validation", "test"]
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    # Collect (path, label) pairs from the fetched dataset.
    paths, labels = [], []
    for category in CATEGORIES:
        folder = SOURCE / category
        if not folder.is_dir():
            sys.exit(f"Missing source folder: {folder}")
        files = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
        )
        paths.extend(files)
        labels.extend([category] * len(files))
    total = len(paths)
    print(f"Found {total} images: {dict(Counter(labels))}")

    # Refuse to write into non-empty target folders to avoid mixing splits.
    for split in SPLITS:
        for category in CATEGORIES:
            target = DATASET / split / category
            if target.exists() and any(target.iterdir()):
                sys.exit(f"{target} is not empty. Clear it before re-running.")

    # Stage 1: carve off the test set, stratified by class.
    n_test = round(total * TEST_FRAC)
    rest_paths, test_paths, rest_labels, test_labels = train_test_split(
        paths, labels, test_size=n_test, stratify=labels, random_state=SEED
    )
    # Stage 2: split the remainder into train and validation, stratified.
    n_val = round(total * VAL_FRAC)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        rest_paths, rest_labels, test_size=n_val, stratify=rest_labels,
        random_state=SEED,
    )

    assignments = {
        "train": zip(train_paths, train_labels),
        "validation": zip(val_paths, val_labels),
        "test": zip(test_paths, test_labels),
    }

    # Copy files into dataset/<split>/<category>/.
    for split, items in assignments.items():
        counts = Counter()
        for src, category in items:
            dst_dir = DATASET / split / category
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)
            counts[category] += 1
        print(f"{split:<10} {sum(counts.values()):>6} -> {dict(counts)}")

    # Sanity check: no image name appears in more than one split.
    seen = {}
    for split in SPLITS:
        for category in CATEGORIES:
            for f in (DATASET / split / category).iterdir():
                if f.name in seen:
                    sys.exit(f"Leak: {f.name} in both {seen[f.name]} and {split}")
                seen[f.name] = split
    print(f"Done. {len(seen)} unique images across all splits, no overlap.")


if __name__ == "__main__":
    main()

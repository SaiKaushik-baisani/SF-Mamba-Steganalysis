"""
Split a directory of cover images into train/val/test by filename, with no
overlap. Run this ONCE before training -- train.py and evaluate.py assume
you're pointing them at already-separated directories, on purpose, so a
train/test leakage bug can't hide inside the training script.

Usage:
    python data/split_dataset.py --cover_dir /path/to/BOSSbase_1.01 \
                                  --out_dir /path/to/split \
                                  --train_frac 0.7 --val_frac 0.15
"""
import argparse
import os
import random
import shutil
import glob


def split_dataset(cover_dir, out_dir, train_frac=0.7, val_frac=0.15, seed=42,
                   extensions=(".pgm", ".png", ".pnm"), copy=False):
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(cover_dir, f"*{ext}")))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No images found in {cover_dir}")

    rng = random.Random(seed)
    rng.shuffle(paths)

    n = len(paths)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    splits = {"train": paths[:n_train], "val": paths[n_train:n_train + n_val],
              "test": paths[n_train + n_val:]}

    for split_name, split_paths in splits.items():
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for src in split_paths:
            dst = os.path.join(split_dir, os.path.basename(src))
            if copy:
                shutil.copy2(src, dst)
            else:
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(src), dst)
        print(f"{split_name}: {len(split_paths)} images -> {split_dir}")
    print("Done. No filename appears in more than one split.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cover_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true")

    args = parser.parse_args()

    split_dataset(
        cover_dir=args.cover_dir,
        out_dir=args.out_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        copy=args.copy,
    )

# Example usage (uncomment and point at your real BOSSbase folder):
# split_dataset(cover_dir="/path/to/BOSSbase_1.01", out_dir="/path/to/split",
#               train_frac=0.7, val_frac=0.15)

"""
On-the-fly cover/stego dataset.

Design choice: generate stego images ON THE FLY at __getitem__ time rather than
pre-generating a fixed stego dataset on disk. Reasons:
  1. Every epoch sees a fresh random embedding realization of the same cover
     image (different random change locations for the same p-map), which acts
     as a free form of data augmentation specific to this task.
  2. You can trivially sweep payload rates / algorithms without regenerating
     and storing multiple copies of the dataset on disk (BOSSbase alone at
     4 payload levels would otherwise mean 5x the storage: 1 cover + 4 stego
     sets).
  3. Matches how SRNet/Zhu-Net and most steganalysis papers actually do it.

Trade-off: this means a fresh HILL cost computation per image per epoch. HILL
is cheap (a few small convolutions), but if you switch to a more expensive
cost function later, consider caching cost maps to disk keyed by image id.

Expected directory layout:
    root/
      cover/
        0001.pgm (or .png)
        0002.pgm
        ...

Payload rate is either fixed or sampled per-item from a provided list, which
lets a single training run cover multiple payloads (helps generalization,
matches "train across payloads" setups common in the literature).
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from src.training.utils import worker_init_fn
from src.steganography.hill import compute_hill_cost
from src.steganography.embedding import simulate_embedding

# (compute_hill_cost / simulate_embedding already defined above in this notebook)


class SteganalysisDataset(Dataset):
    def __init__(
        self,
        cover_dir: str,
        payload_bpps=(0.1, 0.2, 0.3, 0.4),
        image_size: int = 256,
        seed: int = None,
        deterministic: bool = False,
        extensions=(".pgm", ".png", ".pnm"),
    ):
        self.paths = []

        for ext in extensions:
            self.paths.extend(
                sorted(glob.glob(os.path.join(cover_dir, f"*{ext}")))
            )

        if not self.paths:
            raise FileNotFoundError(
                f"No cover images found under {cover_dir} "
                f"with extensions {extensions}."
            )

        self.payload_bpps = list(payload_bpps)
        self.image_size = image_size
        self.deterministic = deterministic
        self.seed = 42 if seed is None else seed
        self.rng = np.random.default_rng(seed)
        self.hill_cache = {}

    def __len__(self):
        # Each source cover contributes two validation samples:
        # its original cover and its corresponding stego.
        if self.deterministic:
            return 2 * len(self.paths)
        return len(self.paths)

    def set_rng(self, rng: np.random.Generator):
        self.rng = rng

    def __getitem__(self, idx):
        if self.deterministic:
            source_idx = idx // 2
            is_stego = (idx % 2 == 1)
        else:
            source_idx = idx
            is_stego = self.rng.random() < 0.5

        img = Image.open(self.paths[source_idx]).convert("L")

        if img.size != (self.image_size, self.image_size):
            img = img.resize(
                (self.image_size, self.image_size),
                Image.BICUBIC,
            )

        cover = np.array(img, dtype=np.uint8)

        if is_stego:
            if self.deterministic:
                bpp = self.payload_bpps[0]
                local_rng = np.random.default_rng(
                    self.seed + source_idx
                )
                rng = local_rng
            else:
                bpp = self.rng.choice(self.payload_bpps)
                rng = self.rng

            if source_idx not in self.hill_cache:
                self.hill_cache[source_idx] = compute_hill_cost(cover)

            rho = self.hill_cache[source_idx]

            image, hard_mask, _, soft_mask = simulate_embedding(
                cover, rho, bpp, rng
            )
            y_det = 1.0

        else:
            image = cover
            hard_mask = np.zeros_like(cover, dtype=np.uint8)
            soft_mask = np.zeros_like(cover, dtype=np.float32)
            y_det = 0.0

        image_t = torch.from_numpy(
            image.astype(np.float32) / 127.5 - 1.0
        ).unsqueeze(0)

        hard_mask_t = torch.from_numpy(
            hard_mask.astype(np.float32)
        ).unsqueeze(0)

        soft_mask_t = torch.from_numpy(
            soft_mask.astype(np.float32)
        ).unsqueeze(0)

        return {
            "image": image_t,
            "y_det": torch.tensor(y_det, dtype=torch.float32),
            "hard_mask": hard_mask_t,
            "soft_mask": soft_mask_t,
            "path": self.paths[source_idx],
        }



def get_dataloader(
    cover_dir,
    payload_bpps=(0.1, 0.2, 0.3, 0.4),
    image_size=256,
    batch_size=16,
    num_workers=4,
    shuffle=True,
    seed=None,
    deterministic=False,
):
    ds = SteganalysisDataset(
        cover_dir,
        payload_bpps=payload_bpps,
        image_size=image_size,
        seed=seed,
        deterministic=deterministic,
    )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        pin_memory=True,
        drop_last=shuffle,
    )


# usage examples further down in this notebook

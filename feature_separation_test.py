import glob
import numpy as np
import torch
from PIL import Image

from src.models.sf_mamba import SFMambaSteganalysis
from src.steganography.hill import compute_hill_cost
from src.steganography.embedding import simulate_embedding


# ============================================================
# CONFIG
# ============================================================

NUM_PAIRS = 20
PAYLOAD = 0.4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# ============================================================
# MODEL
# ============================================================

model = SFMambaSteganalysis().to(DEVICE)
model.eval()

for p in model.parameters():
    p.requires_grad = False


# ============================================================
# DATA
# ============================================================

paths = sorted(
    glob.glob(r"data\BOSSbase\split\train\*.pgm")
)

rng = np.random.default_rng(42)


# ============================================================
# STORAGE
# ============================================================

stage_diffs = {
    "stem": [],
    "local": [],
    "global": [],
    "fusion": [],
    "gap": [],
}


# ============================================================
# PROCESS PAIRED COVER/STego IMAGES
# ============================================================

with torch.no_grad():

    for i, path in enumerate(paths[:NUM_PAIRS]):

        cover = np.array(
            Image.open(path).convert("L"),
            dtype=np.uint8
        )

        rho = compute_hill_cost(cover)

        stego, _, _, _ = simulate_embedding(
            cover,
            rho,
            PAYLOAD,
            rng
        )

        # ----------------------------------------------------
        # INPUT TENSORS
        # ----------------------------------------------------

        cover_t = torch.from_numpy(
            cover.astype(np.float32) / 127.5 - 1.0
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        stego_t = torch.from_numpy(
            stego.astype(np.float32) / 127.5 - 1.0
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # FORWARD COVER
        # ----------------------------------------------------

        def extract_features(x):

            stem = model.stem(x)

            local = model.local_branch(stem)

            global_full = model.global_branch(stem)

            global_feat = torch.nn.functional.interpolate(
                global_full,
                size=local.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            fused = model.fusion(
                local,
                global_feat
            )

            gap = torch.nn.functional.adaptive_avg_pool2d(
                fused,
                1
            ).flatten(1)

            return (
                stem,
                local,
                global_feat,
                fused,
                gap,
            )

        (
            cover_stem,
            cover_local,
            cover_global,
            cover_fusion,
            cover_gap,
        ) = extract_features(cover_t)

        # ----------------------------------------------------
        # FORWARD STEGO
        # ----------------------------------------------------

        (
            stego_stem,
            stego_local,
            stego_global,
            stego_fusion,
            stego_gap,
        ) = extract_features(stego_t)

        # ----------------------------------------------------
        # MEAN ABSOLUTE DIFFERENCE
        # ----------------------------------------------------

        stage_diffs["stem"].append(
            torch.mean(
                torch.abs(stego_stem - cover_stem)
            ).item()
        )

        stage_diffs["local"].append(
            torch.mean(
                torch.abs(stego_local - cover_local)
            ).item()
        )

        stage_diffs["global"].append(
            torch.mean(
                torch.abs(stego_global - cover_global)
            ).item()
        )

        stage_diffs["fusion"].append(
            torch.mean(
                torch.abs(stego_fusion - cover_fusion)
            ).item()
        )

        stage_diffs["gap"].append(
            torch.mean(
                torch.abs(stego_gap - cover_gap)
            ).item()
        )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 65)
print("SF-MAMBA FEATURE SEPARATION")
print("=" * 65)

for stage, values in stage_diffs.items():

    values = np.array(values)

    print(
        f"{stage:10s} | "
        f"mean={values.mean():.8f} | "
        f"median={np.median(values):.8f} | "
        f"min={values.min():.8f} | "
        f"max={values.max():.8f}"
    )

print("=" * 65)
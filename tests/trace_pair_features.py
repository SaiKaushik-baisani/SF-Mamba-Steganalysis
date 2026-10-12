
import torch
from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import SteganalysisDataset

# Set this to the cover-image directory used by your project.
COVER_DIR = r"data\BOSSbase\split\train"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compare(name, a, b):
    diff = (a - b).abs()
    print(
        f"{name:18s} "
        f"shape={tuple(a.shape)} "
        f"mean_abs_diff={diff.mean().item():.8g} "
        f"max_abs_diff={diff.max().item():.8g}"
    )


def main():
    ds = SteganalysisDataset(
        cover_dir=COVER_DIR,
        payload_bpps=(0.1,),
        image_size=256,
        seed=42,
        deterministic=True,
    )

    # Dataset indices 0 and 1 are a matched cover/stego pair.
    cover = ds[0]["image"].unsqueeze(0).to(DEVICE)
    stego = ds[1]["image"].unsqueeze(0).to(DEVICE)

    print("Input difference:")
    compare("input", cover, stego)

    model = SFMambaSteganalysis().to(DEVICE).eval()

    with torch.no_grad():
        def get_features(x):
            stem = model.stem(x)
            local = model.local_branch(stem)
            global_full = model.global_branch(stem)
            global_match = torch.nn.functional.interpolate(
                global_full,
                size=local.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = model.fusion(local, global_match)
            logits = model.detection_head(fused)
            return stem, local, global_full, global_match, fused, logits

        c = get_features(cover)
        s = get_features(stego)

    names = [
        "stem",
        "local",
        "global_full",
        "global_match",
        "fused",
        "detection_logits",
    ]

    print("\nIntermediate differences:")
    for name, a, b in zip(names, c, s):
        compare(name, a, b)

    print("\nFinal logits:")
    print("cover:", c[-1].item())
    print("stego:", s[-1].item())

    
   
    # --------------------------------------------------------
    # SPATIAL-HEAD DIAGNOSTIC
    # Frozen feature extractor, learnable spatial classifier
    # --------------------------------------------------------
    print("\n=== SPATIAL-HEAD DIAGNOSTIC ===")

    cover_features = c[4].detach()
    stego_features = s[4].detach()

    head = torch.nn.Sequential(
        torch.nn.Conv2d(96, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 1),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    features = torch.cat(
        [cover_features, stego_features], dim=0
    )
    labels = torch.tensor([0.0, 1.0], device=DEVICE)

    head.train()
    for step in range(201):
        optimizer.zero_grad()
        logits = head(features).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if step in (0, 1, 5, 10, 25, 50, 100, 200):
            with torch.no_grad():
                probs = torch.sigmoid(head(features).squeeze(-1))
            print(
                f"Step {step:3d} | loss={loss.item():.6f} | "
                f"cover_prob={probs[0].item():.6f} | "
                f"stego_prob={probs[1].item():.6f}"
            )


if __name__ == "__main__":
    main()

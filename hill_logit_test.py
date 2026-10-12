import torch

from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import SteganalysisDataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

dataset = SteganalysisDataset(
    TRAIN_COVER_DIR,
    payload_bpps=[0.4],
    image_size=256,
    seed=42,
)


model = SFMambaSteganalysis().to(DEVICE)

model.eval()


print("Generating HILL pairs...")

for idx in range(10):

    while True:
        cover_sample = dataset[idx]

        if cover_sample["y_det"].item() == 0:
            break

    while True:
        stego_sample = dataset[idx]

        if stego_sample["y_det"].item() == 1:
            break

    cover = cover_sample["image"].unsqueeze(0).to(DEVICE)
    stego = stego_sample["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        cover_logit, _ = model(cover)
        stego_logit, _ = model(stego)

        cover_prob = torch.sigmoid(cover_logit).item()
        stego_prob = torch.sigmoid(stego_logit).item()

    print(
        f"Pair {idx:2d}: "
        f"cover={cover_prob:.6f} "
        f"stego={stego_prob:.6f} "
        f"diff={stego_prob-cover_prob:+.6f}"
    )
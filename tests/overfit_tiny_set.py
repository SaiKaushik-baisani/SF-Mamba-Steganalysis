
import torch
import torch.nn as nn

from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import SteganalysisDataset

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SteganalysisDataset(
    cover_dir=r"data\BOSSbase\split\train",
    payload_bpps=(0.4,),
    image_size=256,
    seed=42,
    deterministic=True,
)

# Use 8 matched cover-stego pairs = 16 images.
indices = list(range(16))
images = torch.stack([dataset[i]["image"] for i in indices]).to(DEVICE)
labels = torch.tensor(
    [dataset[i]["y_det"].item() for i in indices],
    dtype=torch.float32,
    device=DEVICE,
)

model = SFMambaSteganalysis().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

print(f"Device: {DEVICE}")
print(f"Images: {images.shape}")
print(f"Labels: {labels.tolist()}")

for step in range(101):
    model.train()
    optimizer.zero_grad()

    det_logits, _ = model(images)
    loss = criterion(det_logits, labels)

    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        model.eval()
        with torch.no_grad():
            logits, _ = model(images)
            predictions = (logits > 0).float()
            accuracy = (predictions == labels).float().mean().item()

        print(
            f"Step {step:3d} | "
            f"Loss: {loss.item():.6f} | "
            f"Accuracy: {accuracy:.3f}"
        )

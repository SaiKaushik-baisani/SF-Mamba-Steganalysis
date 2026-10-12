
import torch
from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import get_dataloader
from src.training.losses import detection_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = get_dataloader(
    r"data\BOSSbase\split\train",
    payload_bpps=[0.4],
    image_size=256,
    batch_size=2,
    num_workers=0,
    shuffle=False,
    seed=42,
    deterministic=True,
)

batch = next(iter(loader))
x = batch["image"].to(DEVICE)
y = batch["y_det"].to(DEVICE)

print("Labels:", y.cpu().tolist())
print("Image shape:", tuple(x.shape))
print("Image range:", x.min().item(), x.max().item())

model = SFMambaSteganalysis().to(DEVICE).eval()

with torch.no_grad():
    logits, _ = model(x)
    loss = detection_loss(logits, y)

print("Detection logits:", logits.cpu().tolist())
print("Detection probabilities:", torch.sigmoid(logits).cpu().tolist())
print("Detection loss:", loss.item())
print("Parameter count:", sum(p.numel() for p in model.parameters()))

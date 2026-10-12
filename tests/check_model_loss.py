
import torch
from src.models.sf_mamba import SFMambaSteganalysis
from src.training.losses import detection_loss
from src.datasets.dataset import SteganalysisDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SteganalysisDataset(
    r"data\BOSSbase\split\train",
    payload_bpps=[0.1, 0.2, 0.3, 0.4],
    image_size=256,
    seed=42,
    deterministic=True,
)

# One cover and its matching stego image.
cover = dataset[0]
stego = dataset[1]

x = torch.stack([cover["image"], stego["image"]]).to(DEVICE)
y = torch.tensor([0.0, 1.0], device=DEVICE)

model = SFMambaSteganalysis().to(DEVICE)
model.train()

det_logits, loc_logits = model(x)
loss = detection_loss(det_logits, y)

print("Device:", DEVICE)
print("Input shape:", tuple(x.shape))
print("Detection logits shape:", tuple(det_logits.shape))
print("Localization logits shape:", tuple(loc_logits.shape))
print("Detection logits:", det_logits.detach().cpu().tolist())
print("Detection loss:", loss.item())

model.zero_grad(set_to_none=True)
loss.backward()

grad_total = sum(
    p.grad.detach().abs().sum().item()
    for p in model.parameters()
    if p.grad is not None
)

print("Total parameter gradient magnitude:", grad_total)

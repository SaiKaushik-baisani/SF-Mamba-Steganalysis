
import torch
from src.models.sf_mamba import SFMambaSteganalysis
from src.training.losses import detection_loss
from src.datasets.dataset import SteganalysisDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

dataset = SteganalysisDataset(
    r"data\BOSSbase\split\train",
    payload_bpps=[0.1, 0.2, 0.3, 0.4],
    image_size=256,
    seed=42,
    deterministic=True,
)

cover = dataset[0]
stego = dataset[1]

x = torch.stack([cover["image"], stego["image"]]).to(DEVICE)
y = torch.tensor([0.0, 1.0], device=DEVICE)

model = SFMambaSteganalysis().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("Device:", DEVICE)

for step in range(51):
    model.train()
    optimizer.zero_grad()

    det_logits, _ = model(x)
    loss = detection_loss(det_logits, y)
    loss.backward()
    optimizer.step()

    if step in (0, 1, 5, 10, 20, 35, 50):
        model.eval()
        with torch.no_grad():
            logits, _ = model(x)
            current_loss = detection_loss(logits, y).item()
            probs = torch.sigmoid(logits).cpu().tolist()

        print(
            f"Step {step:2d} | loss={current_loss:.6f} | "
            f"cover_prob={probs[0]:.6f} | "
            f"stego_prob={probs[1]:.6f}"
        )

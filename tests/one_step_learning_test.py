
import torch
from src.datasets.dataset import get_dataloader
from src.models.sf_mamba import SFMambaSteganalysis
from src.training.losses import detection_loss

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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

model = SFMambaSteganalysis().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def inspect(stage):
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        loss = detection_loss(logits, y)
        probs = torch.sigmoid(logits)

    print(
        f"{stage}: loss={loss.item():.8f}, "
        f"cover_prob={probs[0].item():.8f}, "
        f"stego_prob={probs[1].item():.8f}"
    )

inspect("Before")

model.train()
optimizer.zero_grad()
logits, _ = model(x)
loss = detection_loss(logits, y)
loss.backward()
optimizer.step()

inspect("After 1 update")


import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets.dataset import SteganalysisDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
PAIRS = 8
STEPS = 100
LR = 1e-3

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Fixed high-pass filter: emphasizes local pixel changes.
        kernel = torch.tensor([
            [0.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 0.0],
        ]).view(1, 1, 3, 3)

        self.register_buffer("highpass", kernel)

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Linear(64, 1)

    def forward(self, x):
        residual = F.conv2d(x, self.highpass, padding=1)
        return self.head(self.features(residual).flatten(1)).squeeze(1)


dataset = SteganalysisDataset(
    r"data\BOSSbase\split\train",
    payload_bpps=[0.4],
    image_size=256,
    seed=SEED,
    deterministic=True,
)

images, labels = [], []

for i in range(PAIRS):
    cover = dataset[2 * i]
    stego = dataset[2 * i + 1]

    images.extend([cover["image"], stego["image"]])
    labels.extend([0.0, 1.0])

x = torch.stack(images).to(DEVICE)
y = torch.tensor(labels, dtype=torch.float32, device=DEVICE)

model = ResidualCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

model.eval()
with torch.no_grad():
    initial_loss = criterion(model(x), y).item()

print("Device:", DEVICE)
print("Fixed samples:", len(y))
print(f"Initial loss: {initial_loss:.6f}")

model.train()

for step in range(1, STEPS + 1):
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    if step == 1:
        grad_total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_total += p.grad.detach().abs().sum().item()

        print(f"Step 1 gradient magnitude: {grad_total:.8f}")

    optimizer.step()

    if step in (1, 10, 25, 50, 75, 100):
        with torch.no_grad():
            logits = model(x)
            current_loss = criterion(logits, y).item()
            if step in (1, 100):
                print(
                    "  logits:",
                    [round(v, 4) for v in logits.cpu().tolist()]
                )
            accuracy = (
                ((logits >= 0) == (y >= 0.5))
                .float()
                .mean()
                .item()
            )

        print(
            f"Step {step:3d} | "
            f"loss={current_loss:.6f} | "
            f"accuracy={accuracy:.3f}"
        )

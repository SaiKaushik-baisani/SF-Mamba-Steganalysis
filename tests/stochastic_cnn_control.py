
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.dataset import SteganalysisDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = r"data\BOSSbase\split\train"
VAL_DIR = r"data\BOSSbase\split\val"

BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.features(x).flatten(1)).squeeze(1)


train_ds = SteganalysisDataset(
    TRAIN_DIR,
    payload_bpps=[0.1, 0.2, 0.3, 0.4],
    image_size=256,
    seed=SEED,
    deterministic=False,
)

val_ds = SteganalysisDataset(
    VAL_DIR,
    payload_bpps=[0.1, 0.2, 0.3, 0.4],
    image_size=256,
    seed=SEED,
    deterministic=True,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model = TinyCNN().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

print("Device:", DEVICE)
print("Train samples:", len(train_ds))
print("Validation samples:", len(val_ds))

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        x = batch["image"].to(DEVICE)
        y = batch["y_det"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * y.size(0)
        train_correct += ((logits >= 0) == (y >= 0.5)).sum().item()
        train_total += y.size(0)

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    cover_probs = []
    stego_probs = []


    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(DEVICE)
            y = batch["y_det"].to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)


            probs = torch.sigmoid(logits)

            cover_probs.extend(probs[y < 0.5].cpu().tolist())
            stego_probs.extend(probs[y >= 0.5].cpu().tolist())

            val_loss_sum += loss.item() * y.size(0)
            val_correct += ((logits >= 0) == (y >= 0.5)).sum().item()
            val_total += y.size(0)

    print(
        f"  Cover mean probability: "
        f"{sum(cover_probs) / max(len(cover_probs), 1):.4f} | "
        f"Stego mean probability: "
        f"{sum(stego_probs) / max(len(stego_probs), 1):.4f}"
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train_loss={train_loss_sum / train_total:.4f} | "
        f"train_acc={train_correct / train_total:.3f} | "
        f"val_loss={val_loss_sum / val_total:.4f} | "
        f"val_acc={val_correct / val_total:.3f}"
    )



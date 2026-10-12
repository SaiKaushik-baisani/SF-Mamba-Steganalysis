
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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


# Eight obvious covers and eight obvious stego samples.
# Covers are all zeros; stego samples are all ones.
x = torch.cat([
    torch.zeros(8, 1, 64, 64),
    torch.ones(8, 1, 64, 64),
]).to(DEVICE)

y = torch.cat([
    torch.zeros(8),
    torch.ones(8),
]).to(DEVICE)

model = TinyCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

print("Device:", DEVICE)

for step in range(101):
    model.train()
    optimizer.zero_grad()

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    if step in (0, 1, 5, 10, 25, 50, 100):
        model.eval()
        with torch.no_grad():
            logits = model(x)
            current_loss = criterion(logits, y).item()
            accuracy = (
                ((logits >= 0) == (y >= 0.5))
                .float().mean().item()
            )

        print(
            f"Step {step:3d} | "
            f"loss={current_loss:.6f} | "
            f"accuracy={accuracy:.3f}"
        )

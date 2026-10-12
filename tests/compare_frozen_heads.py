
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import SteganalysisDataset

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAIRS = 100
STEPS = 100
PAYLOAD = 0.4

def make_dataset(directory):
    return SteganalysisDataset(
        cover_dir=directory,
        payload_bpps=(PAYLOAD,),
        image_size=256,
        seed=42,
        deterministic=True,
    )

train_ds = make_dataset(r"data\BOSSbase\split\train")
val_ds = make_dataset(r"data\BOSSbase\split\val")

model = SFMambaSteganalysis().to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad_(False)

def extract_features(ds):
    features, labels = [], []
    with torch.no_grad():
        for i in range(PAIRS):
            for idx, label in ((2*i, 0.0), (2*i+1, 1.0)):
                x = ds[idx]["image"].unsqueeze(0).to(DEVICE)
                stem = model.stem(x)
                local = model.local_branch(stem)
                glob = model.global_branch(stem)
                glob = F.interpolate(
                    glob, size=local.shape[-2:],
                    mode="bilinear", align_corners=False
                )
                fused = model.fusion(local, glob)
                fused = F.adaptive_avg_pool2d(fused, (32, 32))
                features.append(fused.squeeze(0).cpu())
                labels.append(label)

    return torch.stack(features), torch.tensor(labels)

print("Extracting frozen features...")
train_x, train_y = extract_features(train_ds)
val_x, val_y = extract_features(val_ds)

train_x, train_y = train_x.to(DEVICE), train_y.to(DEVICE)
val_x, val_y = val_x.to(DEVICE), val_y.to(DEVICE)

class GAPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, 1)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1)).squeeze(1)

class SpatialHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def run(name, head):
    head = head.to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(STEPS):
        head.train()
        opt.zero_grad()
        loss = loss_fn(head(train_x), train_y)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        tr_logits = head(train_x)
        va_logits = head(val_x)
        tr_acc = ((tr_logits > 0) == (train_y > 0.5)).float().mean().item()
        va_acc = ((va_logits > 0) == (val_y > 0.5)).float().mean().item()
        va_loss = loss_fn(va_logits, val_y).item()

    print(
        f"{name}: train_acc={tr_acc:.3f}, "
        f"val_acc={va_acc:.3f}, val_loss={va_loss:.6f}"
    )

run("GAP head", GAPHead())
run("Spatial head", SpatialHead())

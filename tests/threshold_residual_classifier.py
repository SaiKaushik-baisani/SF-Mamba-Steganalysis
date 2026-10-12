
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.dataset import SteganalysisDataset

TRAIN_DIR = r"data\BOSSbase\split\train"
VAL_DIR = r"data\BOSSbase\split\val"
PAYLOAD = 0.4
MAX_PAIRS = 500

device = "cuda" if torch.cuda.is_available() else "cpu"

kernel = torch.tensor(
    [[0.0, -1.0, 0.0],
     [-1.0, 4.0, -1.0],
     [0.0, -1.0, 0.0]],
    dtype=torch.float32,
    device=device,
).view(1, 1, 3, 3)


def extract_stats(directory):
    dataset = SteganalysisDataset(
        cover_dir=directory,
        payload_bpps=(PAYLOAD,),
        image_size=256,
        seed=42,
        deterministic=True,
    )

    count = min(MAX_PAIRS, len(dataset) // 2)
    stats, labels = [], []

    for i in range(count):
        for index, label in ((2 * i, 0), (2 * i + 1, 1)):
            image = dataset[index]["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                residual = F.conv2d(image, kernel, padding=1)
                statistic = residual.abs().mean().item()

            stats.append(statistic)
            labels.append(label)

    return np.array(stats), np.array(labels)


train_x, train_y = extract_stats(TRAIN_DIR)
val_x, val_y = extract_stats(VAL_DIR)

# Select the threshold using training data only.
cover_mean = train_x[train_y == 0].mean()
stego_mean = train_x[train_y == 1].mean()
threshold = (cover_mean + stego_mean) / 2

# Determine direction using training data only.
predict_stego_if_high = stego_mean > cover_mean


def evaluate(x, y):
    predictions = (x > threshold) if predict_stego_if_high else (x < threshold)
    accuracy = np.mean(predictions == (y == 1))

    cover = x[y == 0]
    stego = x[y == 1]
    pairwise_auc = np.mean(
        (stego[:, None] > cover[None, :])
        + 0.5 * (stego[:, None] == cover[None, :])
    )

    return accuracy, pairwise_auc, cover.mean(), stego.mean()


print(f"Payload: {PAYLOAD}")
print(f"Training images: {len(train_y)}")
print(f"Validation images: {len(val_y)}")
print(f"Training cover mean: {cover_mean:.8f}")
print(f"Training stego mean: {stego_mean:.8f}")
print(f"Learned threshold: {threshold:.8f}")
print()

for name, x, y in (
    ("TRAIN", train_x, train_y),
    ("VALIDATION", val_x, val_y),
):
    accuracy, auc, c_mean, s_mean = evaluate(x, y)
    print(
        f"{name}: accuracy={accuracy:.4f}, "
        f"AUC={auc:.4f}, "
        f"cover_mean={c_mean:.8f}, "
        f"stego_mean={s_mean:.8f}"
    )

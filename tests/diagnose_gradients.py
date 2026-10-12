import torch

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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

model = SFMambaSteganalysis().to(DEVICE)
model.train()

logits, _ = model(x)
loss = detection_loss(logits, y)
loss.backward()

print("Labels:", y.cpu().tolist())
print("Logits:", logits.detach().cpu().tolist())
print("Loss:", loss.item())
print()

groups = {
    "stem": model.stem,
    "local_branch": model.local_branch,
    "global_branch": model.global_branch,
    "fusion": model.fusion,
    "detection_head": model.detection_head,
}

for name, module in groups.items():
    grad_sum = 0.0
    grad_max = 0.0
    params_with_grad = 0

    for parameter in module.parameters():
        if parameter.grad is not None:
            grad = parameter.grad.detach().abs()
            grad_sum += grad.sum().item()
            grad_max = max(grad_max, grad.max().item())
            params_with_grad += 1

    print(
        f"{name:16s} | "
        f"grad_sum={grad_sum:.8e} | "
        f"grad_max={grad_max:.8e} | "
        f"params_with_grad={params_with_grad}"
    )

import torch
import torch.nn as nn
import numpy as np


def _build_srm_kernels():
    kernels = []

 
    k1 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32)
    k2 = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32)
    k3 = np.array([[0, 0, 0], [0, 1, -2], [0, 0, 1]], dtype=np.float32) 
    k4 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0 
    k5 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32) / 4.0     
    k6 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / 4.0   
    k7 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 4.0  

    base = [k1, k2, k3, k4, k5, k6, k7]


    for k in [k1, k2]:
        for _ in range(3):
            k = np.rot90(k).copy()
            base.append(k.astype(np.float32))

    kernels = np.stack(base, axis=0)  
    return kernels


SRM_KERNELS = _build_srm_kernels()  


class ResidualStem(nn.Module):
    def __init__(self, out_channels: int = 32, learnable: bool = False):
        super().__init__()
        n_kernels = SRM_KERNELS.shape[0]
        weight = torch.from_numpy(SRM_KERNELS).unsqueeze(1)  # (N, 1, 3, 3)

        self.hp_conv = nn.Conv2d(1, n_kernels, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.hp_conv.weight.copy_(weight)
        self.hp_conv.weight.requires_grad = learnable


        self.clip_value = 3.0

        self.project = nn.Sequential(
            nn.Conv2d(n_kernels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        r = self.hp_conv(x)
        r = torch.clamp(r, -self.clip_value, self.clip_value)
        return self.project(r) 



_m = ResidualStem(out_channels=32)
_x = torch.randn(2, 1, 256, 256)
print("ResidualStem output:", _m(_x).shape)

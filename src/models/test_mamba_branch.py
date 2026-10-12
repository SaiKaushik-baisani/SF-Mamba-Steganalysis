"""
Run this BEFORE wiring the new mamba_branch.py into sf_mamba.py.

Checks, mirroring the sanity-check style your own notebook already uses:
  1. parallel_linear_scan matches a naive python-loop reference exactly
     (this re-checks the same thing already verified analytically, but
     now inside real torch/autograd, on your actual machine/GPU).
  2. VSSBlock2D forward pass produces the expected output shape.
  3. Gradients reach every trainable parameter (backward pass works).
  4. Output changes when you scramble the input (i.e. it's not silently
     collapsing to a constant -- a real bug this kind of refactor could
     introduce if a reshape is wrong).

Usage:
    python test_mamba_branch.py
"""

import torch
from mamba_branch import parallel_linear_scan, VSSBlock2D


def reference_loop_scan(A, Bx):
    L = A.shape[1]
    h = torch.zeros_like(Bx[:, 0])
    out = []
    for t in range(L):
        h = A[:, t] * h + Bx[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def check_scan_matches_reference():
    torch.manual_seed(0)
    for shape_extra, L in [((), 37), ((64, 16), 1026), ((8,), 1)]:
        A = torch.rand(2, L, *shape_extra) * 0.9 + 0.05
        Bx = torch.randn(2, L, *shape_extra)
        h_par = parallel_linear_scan(A.clone(), Bx.clone())
        h_ref = reference_loop_scan(A, Bx)
        err = (h_par - h_ref).abs().max().item()
        print(f"  scan check L={L} extra={shape_extra}: max abs err = {err:.2e}")
        assert err < 1e-4, "parallel scan does not match reference loop!"
    print("[OK] parallel_linear_scan matches reference loop scan\n")


def check_vssblock2d():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  using device: {device}")

    model = VSSBlock2D(in_channels=32, hidden_dim=64, pre_downsample=8, n_blocks=4).to(device)
    x = torch.randn(2, 32, 256, 256, device=device, requires_grad=True)

    out = model(x)
    print(f"  input shape:  {tuple(x.shape)}")
    print(f"  output shape: {tuple(out.shape)}")
    assert out.shape == (2, 64, 256, 256), f"unexpected output shape: {out.shape}"
    print("[OK] output shape is correct\n")

    loss = out.sum()
    loss.backward()

    n_params = 0
    n_missing_grad = 0
    for name, p in model.named_parameters():
        n_params += 1
        if p.grad is None or torch.all(p.grad == 0):
            n_missing_grad += 1
            print(f"  [WARN] no/zero gradient for: {name}")
    print(f"  {n_params - n_missing_grad}/{n_params} parameters received a gradient")
    assert x.grad is not None, "gradient did not flow back to the input"
    assert n_missing_grad == 0, "some parameters got no gradient -- check the forward pass"
    print("[OK] gradients flow into every parameter and back to the input\n")

    # Sanity: scrambling the input should change the output (catches a
    # silently-broken reshape that collapses everything to a constant).
    with torch.no_grad():
        x2 = torch.randn(2, 32, 256, 256, device=device)
        out2 = model(x2)
        diff = (out - out2).abs().mean().item()
    print(f"  mean abs diff between two different inputs' outputs: {diff:.4f}")
    assert diff > 1e-4, "output barely changes with a different input -- suspicious"
    print("[OK] output is actually input-dependent\n")


if __name__ == "__main__":
    print("== 1. parallel_linear_scan vs reference loop ==")
    check_scan_matches_reference()
    print("== 2. VSSBlock2D forward + backward + shape checks ==")
    check_vssblock2d()
    print("ALL CHECKS PASSED")
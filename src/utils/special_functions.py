import torch
from torch.autograd import Function
from scipy.special import k0e, k1e

class ScaledBesselK1(Function):
    """
    Exponentially scaled modified Bessel function K of order 1, defined as
    ScaledBesselK1(x) = exp(x) * K_1(x).

    Autograd wrapper for scipy function k1e.
    The backward pass uses the identity: dK1(x)/dx = -K0(x) - K1(x)/x.

    See https://docs.scipy.org/doc/scipy-1.16.0/reference/generated/scipy.special.k1e.html, for more details.
    Note: Should be the same as torch.special.scaled_modified_bessel_k1, but the gradients were unstable in
    the latter.
    """
    @staticmethod
    def forward(ctx, input):
        x_np = input.detach().cpu().numpy()
        scaled_k1 = k1e(x_np)
        ctx.save_for_backward(input)
        return torch.tensor(scaled_k1, dtype=input.dtype, device=input.device)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        input = input.detach()

        x_np = input.cpu().numpy()
        d_scaled_k1 = -k0e(x_np) + k1e(x_np) * (1 - 1 / x_np)

        grad_input = grad_output * torch.tensor(d_scaled_k1, dtype=input.dtype, device=input.device)
        return grad_input

def torch_k1e(x: torch.Tensor) -> torch.Tensor:
    return ScaledBesselK1.apply(x)

def logit(x: torch.Tensor) -> torch.Tensor:
    return torch.log((1 + x) / (1 - x))

def inv_logit(x: torch.Tensor) -> torch.Tensor:
    return (torch.exp(x) - 1.) / (torch.exp(x) + 1.)

import torch
from torch.autograd import Function
from scipy.special import k0e, k1e

class ScaledBesselK1(Function):
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

def torch_k1e(x):
    return ScaledBesselK1.apply(x)

import torch
import numpy as np
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri
from rpy2 import robjects as ro

class Laplace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, par, tmb_model):
        ctx.save_for_backward(par)
        ctx.tmb_model = tmb_model

        par_np = par.detach().numpy()
        with localconverter(ro.default_converter + numpy2ri.converter):
            nll = tmb_model['fn'](par_np)
        return par.new_tensor(nll)

    @staticmethod
    def backward(ctx, grad_output):
        (par,) = ctx.saved_tensors
        tmb_model = ctx.tmb_model

        par_np = par.detach().numpy()
        with localconverter(ro.default_converter + numpy2ri.converter):
            grad_par_np = tmb_model['gr'](par_np)

        grad_par = torch.from_numpy(np.array(grad_par_np)).to(par)
        return grad_output * grad_par, None

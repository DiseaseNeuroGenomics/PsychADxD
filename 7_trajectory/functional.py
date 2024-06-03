# from https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/grad_reverse.py

import torch
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:

        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)

class GradReverseLayer(nn.Module):

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        x = grad_reverse(x, lambd=self.lambd)
        return x



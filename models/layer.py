import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    this should satisfy : [B,S,C] -> [B,S,C]
            
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self,x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 multiple_of: int = 256,
                 bias: bool = False,
                ):
        super().__init__()

        # Start from 4 * dim (standard FFN width).
        # SwiGLU uses three linear layers instead of two,
        # so we reduce to roughly 8/3 * dim to keep a similar parameter budget.
        hidden_dim = int(2*hidden_dim / 3)

        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )
        self.hidden_dim = hidden_dim

        self.gate_proj = nn.Linear(
            dim,
            hidden_dim,
            bias=bias,
        )

        self.up_proj = nn.Linear(
            dim,
            hidden_dim,
            bias=bias,
        )

        self.down_proj = nn.Linear(
            hidden_dim,
            dim,
            bias=bias
        )

    def forward(self, x):


        # gate projection -> SiLU gate
        gate = F.silu(self.gate_proj(x))

        # up projection
        up = self.up_proj(x)

        # elementwise multiply
        x = gate * up

        # down projection
        return self.down_proj(x)

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    y = x * g / rms(x),   rms(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight
    
"""
RMSNorm: This is an alternative of LayerNorm where we nomralize the values by dividing with root mean sqaured norm.
Benefits of RMSNorm over LayerNorm
1) RMSNorm does not require the computation of mean. This saves compute and does not affect performance. 
2) Intuitively, it scales down each dimension to lie on a unit sphere. This type of normalization is very common 
in mathematical problems.

"""

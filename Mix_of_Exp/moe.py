from __future__ import annotations
import torch, torch.nn as nn
from gating import TopKGate
from experts import ExpertMLP

class MoE(nn.Module):
    """Mixture of experts layer (token wise top k routing).
    Implementation based on switch transformers paper and is single GPU friendly.
    """
    def __init__(
        self, dim: int, 
        n_expert: int, 
        k: int = 1, 
        mult: int = 4, 
        swiglu: bool = True, 
        dropout: float = 0.0):
        super().__init__()

        self.dim = dim
        self.n_expert = n_expert
        self.k = k
        self.gate = TopKGate(self.dim, self.n_expert, self.k)
        self.experts = nn.ModuleList(
            [
                ExpertMLP(dim, mult, swiglu, dropout)
            ]
        )

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, C) -> y: (B, T, C), aux_loss
        Steps: flatten tokens along batch and sequence dimension ->
        gate ->
        per-expert forward pass ->
        comppute aux_loss
        """

        B, T, C = x.shape
        S = B * T
        x_flat = x.reshape(S, C)
        idx, w, aux = self.gate(x_flat) # idx shape: (S, k) — for each token, the index of the k chosen experts (integers 0..n_expert-1).
        # w shape: (S, k) — corresponding gating weights for each chosen expert.
        # aux: auxiliary scalar/tensor (e.g., load-balancing/loss from the gate).

        y = torch.zeros_like(x_flat)
        for e in range(self.n_expert):
            for slot in range(self.k):
                sel = (idx[:, slot] == e) # boolean mask of tokens assigned to expert e for slot s, shape (S,)
                if sel.any():
                    x_e = x_flat[sel] # x_e = x_flat[sel] — inputs for that expert, shape (n_sel, C).
                    y_e = self.experts[e](x_e) # y_e = self.expertse — expert output, shape (n_sel, C).
                    y[sel] += w[sel, slot:slot+1] * y_e # y[sel] += w[sel, s:s+1] * y_e — weighted contribution; w slice has shape (n_sel, 1) so it broadcasts over C. Contributions from multiple slots/expert assignments accumulate.
        
        y = y.view(B, T, C)
        return y, aux
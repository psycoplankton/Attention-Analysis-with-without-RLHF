from __future__ import annotations
import torch, torch.nn as nn

class TopKGate(nn.Module):
    """
    Gating mechanism for mixture of experts module. 
    Follows the implementation of Switch Transformers.
    Args:
        dim: hidden dimensions of the token
        n_expert: number of experts
        k: how many experts to choose per token
    Returns:
        (indices, weights, aux_loss) where
        indices -> ids of the experts chosen for each token.
        weights -> the importance(weight) to be given to each expert.
        aux_loss -> auxiliary loss to avoid localized expert assignment.
    """
    def __init__(self, dim: int, n_expert: int, k: int=1):
        super().__init__()
        self.n_expert = n_expert
        self.k = k
        self.w_g = nn.Linear(dim, n_expert, bias=True)

    def forward(self, x: torch.Tensor):
        # shape of x -> (S, D) where S is batch*seq, D is dim
        logits = self.w_g(x) #(S, E)
        probs = torch.softmax(logits) # (S, E)
        topk_vals, topk_idx = torch.topk(probs, k=self.k, dim=-1)

        # Load balancing auxiliary loss
        S, E = probs.shape
        importance = probs.mean(dim=0) #Avg importance of each expert per token
        hard1 = topk_idx[:, 0] # the top expert id for each token
        load = torch.zeros(E, device=x.device) # to count how many tokens were routed to each expert
        load.scatter_add_(0, hard1, torch.ones_like(hard1, dtype=load.dtype)) # load[e] will now tell the number of token routed to expert e.
        load = load / max(S, 1)
        aux_loss = (E * (importance * load).sum())

        return topk_idx, topk_vals, aux_loss
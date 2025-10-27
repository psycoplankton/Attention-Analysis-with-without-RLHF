from __future__ import annotations
import torch, torch.nn as nn

class RewardModel(nn.Module):
    """Transformer encoder -> pooled representation -> sclar reward.
    Bidirectional encoder is fine for reward modeling (not used for generation)."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int, 
        n_layer: int = 4, 
        n_head: int = 4,
        n_embed: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=n_embed,
            nhead=n_head,
            dim_feedforward=4*n_embed,
            dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, 1)

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0) # to include batch
        h = self.tok_emb(x) + self.pos_emb(pos)
        pad_mask = (x == 2)
        h = self.encoder(h, src_key_padding_mask=pad_mask) # The src_key_padding_mask tells the Transformer which time-step positions are padding and should be ignored by attention.
        h = self.ln(h) # (B, T, n_embed) 
        mask = (~pad_mask).float().unsqueeze(-1) # sequence positions in time to all the padded tokens
        h_sum = (h*mask).sum(dim=1) #convert all the pad ids' embeddings to 0 -> (B, n_embed) 
        len_ = mask.sum(dim=1).clamp_min(1.0) # Sum the rewards over 1st(token) dimension. which will then represent the rewards for each dimension.
        # (8, 256) -> (8, 1) (storing the count of true values)
        pooled = h_sum/len_ # Computes the mean embedding per sequence by dividing the summed embeddings by the valid token counts → fixed-size pooled representation (B, n_embed).
        r =self.head(pooled).squeeze(-1) # Passes the pooled vector through a linear layer that maps n_embed → 1, producing a scalar reward per example (B, 1), then `squeeze(-1)` yields shape (B,).
        return r
        
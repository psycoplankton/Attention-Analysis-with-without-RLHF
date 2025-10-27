from __future__ import annotations
import torch
from typing import List, Tuple

import sys
try:
    from LLM_parts.tokenizer_bpe import BPETokenizer
    _HAS_BPE = True
except Exception:
    _HAS_BPE = False
    print(f"Warning: BPETokenizer not found.")

from SFT.formatters import Example, format_example, format_prompt_only

# ---------- tokenizer helpers ----------
class RLHFTokenizer:
    def __init__(self, block_size: int, bpe_dir: str | None = None, vocab_size: int = 8000):
        self.block_size = block_size
        self.tok = None
        if _HAS_BPE:
            try:
                self.tok = BPETokenizer(vocab_size=vocab_size)
                if bpe_dir:
                    self.tok.load(bpe_dir)
            except Exception:
                self.tok = None
        if self.tok is None and ByteTokenizer is not None:
            self.tok = ByteTokenizer()
        if self.tok is None:
            raise RuntimeError("No tokenizer available for RLHF.")

    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, 'vocab_size', 256)

    def encode(self, text: str) -> List[int]:
        ids = self.tok.encode(text)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ids

    def decode(self, ids: List[int]) -> str:
        if hasattr(self.tok, 'decode'):
            return self.tok.decode(ids)
        return bytes(ids).decode('utf-8', errors='ignore')

# ---------- logprob utilities ----------
def shift_labels(x: torch.Tensor) -> torch.Tensor:
    # For causal LM: predict x[t+1] from x[:t]
    return x[:, 1:].contiguous() #Shift the labels by one position

def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token logprobs of the given labels.
    logits: (B,T,V), labels: (B,T) over same T
    returns: (B,T) log p(labels)

    gather function uses the index tensor passed to it as a registry
    to pick out the tokens from the original index which is at position as given by index tensor.
    """
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

@torch.no_grad()
def model_logprobs(model, x: torch.Tensor) -> torch.Tensor:
    # compute log p(x[t+1] | x[:t]) for t
    logits, _, _ = model.lm(x, None) if hasattr(model, 'lm') else model(x, None)
    labels = shift_labels(x)
    lp = gather_logprobs(logits[:, :-1, :], labels) #gather the logprobs of corresponding to the labels.
    return lp  # (B, T-1)

# ---------- KL ----------

def approx_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    # Mean over tokens: KL(pi||ref) ≈ (logp_pi - logp_ref).mean()
    return (policy_logp - ref_logp).mean()

# ---------- small prompt source ----------
try:
    from datasets import load_dataset as _load_ds
except Exception:
    _load_ds = None

def sample_prompts(n: int) -> List[str]:
    if _load_ds is not None:
        try:
            ds = _load_ds("tatsu-lab/alpaca", split="train[:24]")
            arr = []
            for r in ds:
                inst = (r.get('instruction') or '').strip()
                inp = (r.get('input') or '').strip()
                if inp:
                    inst = inst + "\n" + inp
                if inst:
                    arr.append(inst)
                if len(arr) >= n:
                    break
            if arr:
                return arr
        except Exception:
            pass
    # fallback
    base = [
        "Explain the purpose of attention in transformers.",
        "Give two pros and cons of BPE tokenization.",
        "Summarize why PPO is used in RLHF.",
        "Write a tiny Python function that reverses a list.",
    ]
    return (base * ((n+len(base)-1)//len(base)))[:n] # In case we need to sample more prompts than are available. We repeat them.
from __future__ import annotations
from typing import List, Tuple
import torch
import traceback

import sys
from pathlib import Path as _P

try:
    from LLM_parts.tokenizer import BPETokenizer
    _HAS_BPE = True
except Exception:
    _HAS_BPE = False
try:
    from LLM_parts.tokenizer import ByteTokenizer
except Exception:
    ByteTokenizer = None

from formatters import Example, format_example, format_prompt_only

class SFTCollator:
    """Turn (instruction,response) into token ids and masked labels for causal LM (6.2).
    Labels for the prompt part are set to -100 so they don't contribute to loss.
    """
    def __init__(self, block_size: int = 256, bpe_dir: str | None = None):
        self.block_size = block_size
        self.tok = None
        if _HAS_BPE:
            # If a trained tokenizer directory exists from Part 4, you can `load` it.
            # Otherwise we create an ad-hoc BPE on the fly using fallback prompts during demo.
            try:
                self.tok = BPETokenizer(vocab_size=8000)
                if bpe_dir:
                    self.tok.load(bpe_dir)
                    print(f"Loaded BPE tokenizer from {bpe_dir}")
                else:
                    # weak ad-hoc training would belong elsewhere; for the demo we assume Part 4 tokenizer exists
                    pass
            except Exception:
                print(traceback.format_exc())
                self.tok = None
        if self.tok is None and ByteTokenizer is not None:
            self.tok = ByteTokenizer()
        if self.tok is None:
            raise RuntimeError("No tokenizer available. Install tokenizers or ensure Part 3 ByteTokenizer exists.")

    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, 'vocab_size', 256)
    
    def encode(self, text: str) -> List[int]:
        if hasattr(self.tok, 'encode'):
            ids = self.tok.encode(text)
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return ids
        return list(text.encode('utf-8'))

    def collate(self, batch: List[Tuple[str, str]]):
        # Build "prompt + response" and create label mask where prompt positions are -100.
        input_ids = []
        labels = []
        for prompt, response in batch:
            prefix_text = format_prompt_only(prompt).replace('</s>','') #this will be passed to the model for generation
            text = format_example(Example(prompt, response)) # this is the complete text
            ids = self.encode(text)[:self.block_size]
            prompt_ids = self.encode(prefix_text)[:self.block_size] # ids of the tokens that are present in prompt.
            n_prompt = min(len(prompt_ids), len(ids)) #Take either the ids that can be accomocated (block_size) or the complete prompt ids.
            x = ids 
            y = ids.copy()
            for t in range(len(y) - 1):
                y[t] = ids[t + 1] # making the input-target type data.
            y[-1] = -100
            for i in range(n_prompt-1):
                y[i] = -100 # Mark all the prompt tokens to be -100. so that they are not considered while calculating loss. 
            input_ids.append(x)
            labels.append(y) 

            # pad to block_size
        def pad_to(ids, val):
            if len(ids) < self.block_size:
                ids = ids + [val]*(self.block_size - len(ids))
            return ids[:self.block_size]
        x = torch.tensor([pad_to(s, 2) for s in input_ids], dtype=torch.long)
        y = torch.tensor([pad_to(s, -100) for s in labels], dtype=torch.long)
        return x, y
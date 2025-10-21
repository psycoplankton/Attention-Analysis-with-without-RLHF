from __future__ import annotations
from typing import List

class LengthCurriculum:
    """6.3 Curriculum: iterate examples from short→long prompts (one pass demo)."""
    def __init__(self, items: List[tuple[str,str]]):
        self.items = sorted(items, key=lambda p: len(p[0])) # sort examples based on length of prompt.
        self._i = 0
    def __iter__(self):
        self._i = 0
        return self
    def __next__(self):
        if self._i >= len(self.items):
            raise StopIteration # If the iterator has crossed the length of the dataset, then stop iteration.
        it = self.items[self._i]
        self._i += 1
        return it
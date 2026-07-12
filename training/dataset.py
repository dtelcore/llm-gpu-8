"""
training/dataset.py

Turns the corpus (list of sentence strings) into a contiguous token stream
and yields (X, Y) sliding-window batches, Y being X shifted by one position
(next-character prediction).
"""

from typing import Iterator, List, Tuple

import numpy as np


class WindowedDataset:
    def __init__(self, corpus: List[str], tokenizer, max_len: int, batch_size: int) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size

        joined_text = " ".join(corpus)
        self.tokens = np.array(tokenizer.encode(joined_text), dtype=np.int64)
        self.total_tokens = len(self.tokens)

        if self.total_tokens < max_len + 1:
            raise ValueError(
                f"Corpus has only {self.total_tokens} tokens, need at least "
                f"{max_len + 1} for a single window (max_len={max_len})."
            )

    def num_windows(self) -> int:
        return self.total_tokens - self.max_len

    def num_batches(self) -> int:
        return max(1, self.num_windows() // self.batch_size)

    def iter_batches(self, shuffle: bool = True, rng: np.random.Generator = None) -> Iterator[List[Tuple[np.ndarray, np.ndarray]]]:
        """Yields lists of (x, y) sequence pairs, one list per mini-batch."""
        rng = rng or np.random.default_rng()
        starts = np.arange(self.num_windows())
        if shuffle:
            rng.shuffle(starts)

        for b in range(self.num_batches()):
            batch_starts = starts[b * self.batch_size : (b + 1) * self.batch_size]
            if len(batch_starts) == 0:
                continue
            pairs = []
            for start in batch_starts:
                x = self.tokens[start : start + self.max_len]
                y = self.tokens[start + 1 : start + self.max_len + 1]
                pairs.append((x, y))
            yield pairs

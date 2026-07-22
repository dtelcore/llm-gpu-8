"""
tokenizer/bpe.py

Simple byte-pair encoding tokenizer for Stage 3.3 experiments.
Does not replace CharacterGPTTokenizer as the BiggerTest default.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union


def _word_tokens(text: str) -> List[str]:
    """Split on whitespace; keep spaces as separate tokens so decode round-trips."""
    parts: List[str] = []
    buf = []
    for ch in text:
        if ch.isspace():
            if buf:
                parts.append("".join(buf))
                buf = []
            parts.append(ch)
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def _get_stats(seqs: List[List[str]]) -> Counter:
    stats: Counter = Counter()
    for seq in seqs:
        for i in range(len(seq) - 1):
            stats[(seq[i], seq[i + 1])] += 1
    return stats


def _merge_pair(seqs: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
    a, b = pair
    merged = a + b
    out: List[List[str]] = []
    for seq in seqs:
        new_seq: List[str] = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        out.append(new_seq)
    return out


class BPETokenizer:
    """Whitespace-aware BPE over character symbols (experiment path)."""

    def __init__(self) -> None:
        self.vocab: List[str] = []
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0

    @classmethod
    def from_corpus(
        cls,
        corpus: Iterable[str],
        num_merges: int = 200,
        max_chars: Optional[int] = 200_000,
    ) -> "BPETokenizer":
        text = " ".join(corpus)
        if max_chars is not None:
            text = text[: int(max_chars)]
        inst = cls()
        inst.train(text, num_merges=num_merges)
        return inst

    def train(self, text: str, num_merges: int = 200) -> None:
        # Start from characters present in text (+ keep space/newline as atoms).
        chars = sorted(set(text))
        seqs: List[List[str]] = []
        for word in _word_tokens(text):
            seqs.append(list(word))

        merges: List[Tuple[str, str]] = []
        vocab_set = set(chars)
        for _ in range(int(num_merges)):
            stats = _get_stats(seqs)
            if not stats:
                break
            pair, _count = stats.most_common(1)[0]
            if _count < 2:
                break
            merges.append(pair)
            vocab_set.add(pair[0] + pair[1])
            seqs = _merge_pair(seqs, pair)

        self.merges = merges
        self.vocab = sorted(vocab_set, key=lambda s: (len(s), s))
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size = len(self.vocab)

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        seq = list(symbols)
        for a, b in self.merges:
            merged = a + b
            new_seq: List[str] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
        return seq

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in _word_tokens(text):
            pieces = self._apply_merges(list(word))
            for p in pieces:
                tid = self.token_to_id.get(p)
                if tid is not None:
                    ids.append(tid)
                else:
                    # Fall back to characters if an unknown merge piece appears
                    for ch in p:
                        if ch in self.token_to_id:
                            ids.append(self.token_to_id[ch])
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_token.get(i, "") for i in ids)

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "type": "bpe",
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
        }
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BPETokenizer":
        state = json.loads(Path(filepath).read_text(encoding="utf-8"))
        inst = cls()
        inst.vocab = state["vocab"]
        inst.merges = [tuple(p) for p in state["merges"]]
        inst.token_to_id = state["token_to_id"]
        inst.id_to_token = {int(k): v for k, v in state["id_to_token"].items()}
        inst.vocab_size = len(inst.vocab)
        return inst

    def coverage_stats(self, text: str, context_tokens: int) -> Dict[str, float]:
        """Compare char vs BPE semantic span inside a fixed token window."""
        char_len = len(text)
        ids = self.encode(text)
        window = ids[:context_tokens]
        decoded = self.decode(window)
        return {
            "char_len": float(char_len),
            "bpe_token_count": float(len(ids)),
            "chars_per_token": float(char_len) / max(1, len(ids)),
            "context_token_window": float(context_tokens),
            "chars_covered_in_window": float(len(decoded)),
            "compression_vs_chars": float(char_len) / max(1, len(ids)),
        }

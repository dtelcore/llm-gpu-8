"""
tokenizer/tokenizer.py

Character-level tokenizer for the Kepler GT 730 GPT stack.

Vocabulary rule matches setup/training_setup.py exactly:
    vocab_size = len(set(' '.join(corpus)))
so a tokenizer built from the same corpus always agrees with the
`vocab_size` recorded in training_config.json.
"""

import json
from pathlib import Path
from typing import Dict, List, Union

from logging_config import logger


class CharacterGPTTokenizer:
    """Maps characters <-> integer ids for a fixed corpus vocabulary."""

    def __init__(self) -> None:
        self.vocab: List[str] = []
        self.vocab_size: int = 0
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

    @classmethod
    def from_corpus(cls, corpus: List[str]) -> "CharacterGPTTokenizer":
        instance = cls()
        instance.build_vocab(corpus)
        return instance

    def build_vocab(self, corpus: List[str]) -> None:
        """Extract unique characters from the corpus (joined with spaces)."""
        full_text = " ".join(corpus)
        self.vocab = sorted(set(full_text))
        self.vocab_size = len(self.vocab)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        logger.info(f"Tokenizer vocab built: {self.vocab_size} unique characters")

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_char.get(i, "") for i in ids)

    def id_to_token(self, token_id: int) -> str:
        return self.id_to_char.get(token_id, "<unk>")

    def token_to_id(self, token: str) -> int:
        return self.char_to_id.get(token, -1)

    def save_vocab(self, filepath: Union[str, Path]) -> None:
        state = {
            "vocab": self.vocab,
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.info(f"Tokenizer vocab saved to {filepath}")

    @classmethod
    def load_vocab(cls, filepath: Union[str, Path]) -> "CharacterGPTTokenizer":
        instance = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        instance.vocab = state["vocab"]
        instance.vocab_size = len(instance.vocab)
        instance.char_to_id = state["char_to_id"]
        instance.id_to_char = {int(k): v for k, v in state["id_to_char"].items()}
        logger.info(f"Tokenizer vocab loaded from {filepath} ({instance.vocab_size} chars)")
        return instance

    def trace_round_trip(self, text: str) -> None:
        """Print char -> id -> char for every character in `text` (debug aid)."""
        print(f"\n[TraceTokens] Round-trip for: {text!r}")
        print(f"{'Char':<8} | {'ID':<5} | Decoded")
        print("-" * 32)
        for char in text:
            tid = self.char_to_id.get(char, -1)
            decoded = self.id_to_char.get(tid, "<unk>")
            print(f"{char!r:<8} | {tid:<5} | {decoded!r}")

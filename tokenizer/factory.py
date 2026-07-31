"""
tokenizer/factory.py

Build / save / load char or BPE tokenizers. New training defaults to BPE.
Legacy char vocab.json (no "type" field) still loads for old checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Union

from logging_config import logger
from tokenizer.bpe import BPETokenizer
from tokenizer.tokenizer import CharacterGPTTokenizer

DEFAULT_TOKENIZER = "bpe"
DEFAULT_BPE_MERGES = 200

Tokenizer = Union[CharacterGPTTokenizer, BPETokenizer]


def build_tokenizer(
    corpus: Iterable[str],
    kind: str = DEFAULT_TOKENIZER,
    num_merges: int = DEFAULT_BPE_MERGES,
) -> Tokenizer:
    kind = (kind or DEFAULT_TOKENIZER).strip().lower()
    corpus_list = list(corpus)
    if kind == "char":
        return CharacterGPTTokenizer.from_corpus(corpus_list)
    if kind == "bpe":
        tok = BPETokenizer.from_corpus(corpus_list, num_merges=int(num_merges))
        logger.info(
            "BPE tokenizer built: vocab_size=%s merges=%s",
            tok.vocab_size, len(tok.merges),
        )
        return tok
    raise ValueError(f"Unknown tokenizer kind {kind!r}; expected 'bpe' or 'char'")


def save_tokenizer(tokenizer: Tokenizer, filepath: Union[str, Path]) -> None:
    tokenizer.save_vocab(filepath)


def load_tokenizer(filepath: Union[str, Path]) -> Tokenizer:
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    if state.get("type") == "bpe" or "merges" in state:
        tok = BPETokenizer.load(path)
        logger.info("BPE tokenizer loaded from %s (%s tokens)", path, tok.vocab_size)
        return tok
    return CharacterGPTTokenizer.load_vocab(path)


def tokenizer_kind_from_config(dataset: Optional[dict], args=None) -> str:
    if args is not None and getattr(args, "tokenizer", None):
        return str(args.tokenizer).strip().lower()
    if dataset and dataset.get("tokenizer"):
        return str(dataset["tokenizer"]).strip().lower()
    return DEFAULT_TOKENIZER


def bpe_merges_from_config(dataset: Optional[dict], args=None) -> int:
    if args is not None and getattr(args, "bpe_merges", None) is not None:
        return int(args.bpe_merges)
    if dataset and dataset.get("bpe_merges") is not None:
        return int(dataset["bpe_merges"])
    return DEFAULT_BPE_MERGES

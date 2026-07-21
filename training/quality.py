"""
training/quality.py

Heuristic generation-quality scores (spelling, punctuation, grammar, semantics)
and sequential inter-quarter comparison / best promotion.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from logging_config import logger
from paths import DATA_DIR, list_quarter_dirs, run_root_for_checkpoint
from training.checkpoint import load_checkpoint, promote_best

DEFAULT_QUALITY_PROMPT = "once upon a"
DEFAULT_QUALITY_MAX_NEW_TOKENS = 256
DEFAULT_QUALITY_TEMPERATURE = 0.6
DEFAULT_QUALITY_TOP_K = 10
DEFAULT_QUALITY_TOP_P = 0.9

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENT_END_RE = re.compile(r"[.!?]+")
_WORDLIST_CACHE: Optional[set] = None


@dataclass
class QualityScores:
    spelling: float
    punctuation: float
    grammar: float
    semantics: float
    aggregate: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _load_wordlist() -> Optional[set]:
    global _WORDLIST_CACHE
    if _WORDLIST_CACHE is not None:
        return _WORDLIST_CACHE or None
    candidates = [
        DATA_DIR / "wordlist.txt",
        DATA_DIR / "words.txt",
        DATA_DIR / "english_words.txt",
    ]
    for path in candidates:
        if path.exists():
            words = set()
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    w = line.strip().lower()
                    if w and w.isalpha():
                        words.add(w)
            _WORDLIST_CACHE = words
            return words
    _WORDLIST_CACHE = set()
    return None


def _tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def score_spelling(text: str) -> float:
    """Higher = fewer garbage/repeated-char tokens; optional wordlist boost."""
    words = _tokenize_words(text)
    if not words:
        return 0.0

    garbage = 0
    for w in words:
        lower = w.lower()
        if len(lower) >= 4 and len(set(lower)) == 1:
            garbage += 1
            continue
        # long runs of the same char (e.g. "helllllo")
        if re.search(r"(.)\1{3,}", lower):
            garbage += 1

    clean_ratio = 1.0 - (garbage / len(words))

    wordlist = _load_wordlist()
    if wordlist:
        known = sum(1 for w in words if w.lower() in wordlist)
        known_ratio = known / len(words)
        return _clamp01(0.4 * clean_ratio + 0.6 * known_ratio)

    # Without a wordlist: prefer alphabetic tokens over digit/symbol soup.
    alpha_chars = sum(1 for c in text if c.isalpha())
    total = max(1, len(text.strip()))
    alpha_ratio = alpha_chars / total
    return _clamp01(0.7 * clean_ratio + 0.3 * alpha_ratio)


def score_punctuation(text: str) -> float:
    """Sentence terminators, spacing around punctuation, quote/paren balance."""
    stripped = text.strip()
    if not stripped:
        return 0.0

    has_terminator = 1.0 if _SENT_END_RE.search(stripped) else 0.0
    # Prefer ending with terminator for multi-word output
    ends_ok = 1.0 if stripped[-1] in ".!?" else (0.5 if len(stripped.split()) < 4 else 0.0)

    # Bad spacing: "word ," or " ." or double spaces around punct
    bad_space = len(re.findall(r"\s+[,.!?;:]", stripped)) + len(re.findall(r"[,.!?;:]{2,}", stripped))
    space_score = _clamp01(1.0 - bad_space / max(1, len(stripped) // 20))

    # Balance quotes / parens
    balance = 1.0
    for open_c, close_c in (('"', '"'), ("'", "'"), ("(", ")"), ("[", "]")):
        if open_c == close_c:
            if stripped.count(open_c) % 2 != 0:
                balance -= 0.15
        else:
            if stripped.count(open_c) != stripped.count(close_c):
                balance -= 0.15
    balance = _clamp01(balance)

    return _clamp01(0.35 * has_terminator + 0.25 * ends_ok + 0.25 * space_score + 0.15 * balance)


def score_grammar(text: str) -> float:
    """Capitalization after terminators, extreme repetition, broken spacing."""
    stripped = text.strip()
    if not stripped:
        return 0.0

    # Capitalization after sentence end
    caps_ok = 0
    caps_total = 0
    for m in _SENT_END_RE.finditer(stripped):
        rest = stripped[m.end() :].lstrip()
        if not rest:
            continue
        caps_total += 1
        if rest[0].isupper():
            caps_ok += 1
    caps_score = (caps_ok / caps_total) if caps_total else (1.0 if stripped[0].isupper() else 0.5)

    words = _tokenize_words(stripped)
    if words:
        # Extreme consecutive word repetition
        reps = 0
        for i in range(1, len(words)):
            if words[i].lower() == words[i - 1].lower():
                reps += 1
        rep_score = _clamp01(1.0 - reps / len(words))
    else:
        rep_score = 0.0

    # Broken spacing (multiple spaces, space at start of "word")
    multi_space = len(re.findall(r"  +", stripped))
    space_score = _clamp01(1.0 - multi_space / max(1, len(stripped) // 30))

    return _clamp01(0.4 * caps_score + 0.35 * rep_score + 0.25 * space_score)


def score_semantics(text: str, prompt: str = "") -> float:
    """Prompt-token overlap, unique-token ratio, non-gibberish entropy."""
    stripped = text.strip()
    if not stripped:
        return 0.0

    # Continuation = text after prompt if prompt is a prefix
    continuation = stripped
    if prompt and stripped.lower().startswith(prompt.lower()):
        continuation = stripped[len(prompt) :].lstrip()

    cont_words = [w.lower() for w in _tokenize_words(continuation)]
    prompt_words = [w.lower() for w in _tokenize_words(prompt)]

    if cont_words:
        unique_ratio = len(set(cont_words)) / len(cont_words)
    else:
        unique_ratio = 0.0

    # Soft prompt stickiness: share of prompt content words that reappear
    if prompt_words and cont_words:
        prompt_set = set(prompt_words)
        overlap = sum(1 for w in cont_words if w in prompt_set) / len(cont_words)
        # Prefer some topical glue but not parroting the whole prompt
        stickiness = 1.0 - abs(overlap - 0.15) / 0.85
        stickiness = _clamp01(stickiness)
    else:
        stickiness = 0.5

    # Character entropy (gibberish tends toward extreme low or high for short noise)
    chars = continuation.lower() if continuation else stripped.lower()
    if chars:
        freq: Dict[str, int] = {}
        for c in chars:
            freq[c] = freq.get(c, 0) + 1
        ent = 0.0
        n = len(chars)
        for count in freq.values():
            p = count / n
            ent -= p * math.log2(p)
        # English-ish text often ~3–4.5 bits; map into 0–1 softly
        entropy_score = _clamp01((ent - 1.5) / 3.0)
    else:
        entropy_score = 0.0

    return _clamp01(0.35 * unique_ratio + 0.30 * stickiness + 0.35 * entropy_score)


def score_generation(
    text: str,
    prompt: str = "",
    *,
    weights: Optional[Dict[str, float]] = None,
) -> QualityScores:
    """Score text on spelling/punctuation/grammar/semantics; aggregate weighted mean."""
    w = {
        "spelling": 1.0,
        "punctuation": 1.0,
        "grammar": 1.0,
        "semantics": 1.0,
    }
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})

    spelling = score_spelling(text)
    punctuation = score_punctuation(text)
    grammar = score_grammar(text)
    semantics = score_semantics(text, prompt=prompt)

    total_w = sum(w.values()) or 1.0
    aggregate = (
        w["spelling"] * spelling
        + w["punctuation"] * punctuation
        + w["grammar"] * grammar
        + w["semantics"] * semantics
    ) / total_w

    return QualityScores(
        spelling=_clamp01(spelling),
        punctuation=_clamp01(punctuation),
        grammar=_clamp01(grammar),
        semantics=_clamp01(semantics),
        aggregate=_clamp01(aggregate),
    )


def _delta_label(curr: float, prev: Optional[float]) -> str:
    if prev is None:
        return "baseline"
    diff = curr - prev
    if abs(diff) < 0.02:
        return "stable"
    return "improving" if diff > 0 else "regressing"


def compare_quarters(
    run_dir: str,
    *,
    prompt: str = DEFAULT_QUALITY_PROMPT,
    max_new_tokens: int = DEFAULT_QUALITY_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_QUALITY_TEMPERATURE,
    top_k: Optional[int] = DEFAULT_QUALITY_TOP_K,
    top_p: Optional[float] = DEFAULT_QUALITY_TOP_P,
    seed: int = 42,
    weights: Optional[Dict[str, float]] = None,
    interactive_promote: bool = True,
    set_best: Optional[str] = None,
) -> List[Dict]:
    """Sequential generate+score across quarter_* checkpoints; optionally promote best."""
    from model.gpt import GPTModel

    root = run_root_for_checkpoint(run_dir)
    quarters = list_quarter_dirs(root)
    if not quarters:
        print(f"No quarter_* checkpoints found under {root}")
        logger.warning("compare_quarters: no quarters under %s", root)
        return []

    results: List[Dict] = []
    prev_agg: Optional[float] = None

    print("=" * 70)
    print(f"QUALITY TRIAL — sequential generation across quarters in {root}")
    print(f"prompt={prompt!r} | tokens={max_new_tokens} | temp={temperature} | top_k={top_k} | top_p={top_p}")
    print("=" * 70)

    for qdir in quarters:
        gpt_config, params, tokenizer, _, state = load_checkpoint(str(qdir))
        model = GPTModel(gpt_config, params)
        step = int(state.get("step", 0))

        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            print(f"[{qdir.name}] prompt encodes empty; skipping")
            continue

        rng = np.random.default_rng(seed)
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng=rng,
        )
        text = tokenizer.decode(generated_ids)
        scores = score_generation(text, prompt=prompt, weights=weights)
        trend = _delta_label(scores.aggregate, prev_agg)
        prev_agg = scores.aggregate

        row = {
            "name": qdir.name,
            "path": str(qdir),
            "step": step,
            "text": text,
            "scores": scores.as_dict(),
            "trend": trend,
        }
        results.append(row)

        print("\n" + "-" * 70)
        print(f"{qdir.name} | step={step:,} | aggregate={scores.aggregate:.3f} ({trend})")
        print(
            f"  spelling={scores.spelling:.3f}  punctuation={scores.punctuation:.3f}  "
            f"grammar={scores.grammar:.3f}  semantics={scores.semantics:.3f}"
        )
        print(text[:500] + ("…" if len(text) > 500 else ""))
        logger.info(
            "[quality] quarter=%s step=%s aggregate=%.4f spelling=%.4f punctuation=%.4f "
            "grammar=%.4f semantics=%.4f trend=%s",
            qdir.name, step, scores.aggregate, scores.spelling, scores.punctuation,
            scores.grammar, scores.semantics, trend,
        )

    if not results:
        return results

    if set_best:
        chosen = _resolve_set_best(root, set_best, results)
        if chosen:
            _do_promote(root, chosen, results)
        return results

    if interactive_promote:
        _prompt_promote(root, results)

    return results


def _resolve_set_best(root: Path, set_best: str, results: Sequence[Dict]) -> Optional[Path]:
    name = set_best.strip()
    by_name = {r["name"]: Path(r["path"]) for r in results}
    if name in by_name:
        return by_name[name]
    candidate = Path(name)
    if not candidate.is_absolute():
        candidate = root / name
    if (candidate / "config.json").exists():
        return candidate
    print(f"--set-best '{set_best}' not found among quarters; skipping promote")
    return None


def _do_promote(root: Path, source: Path, results: Sequence[Dict]) -> None:
    match = next((r for r in results if Path(r["path"]) == source or r["name"] == source.name), None)
    meta = {
        "step": match["step"] if match else None,
        "scores": match["scores"] if match else None,
        "trend": match["trend"] if match else None,
    }
    best = promote_best(root, source, meta=meta)
    print(f"\nPromoted '{source.name}' -> {best}")


def _prompt_promote(root: Path, results: Sequence[Dict]) -> None:
    print("\n" + "=" * 70)
    print("Promote one quarter as best/? (or Enter to skip)")
    for i, r in enumerate(results, 1):
        s = r["scores"]
        print(f"  {i}. {r['name']} (step={r['step']:,}, agg={s['aggregate']:.3f}, {r['trend']})")
    try:
        choice = input("Select number to promote [default=skip]: ").strip()
    except EOFError:
        return
    if not choice:
        print("Skipped best promotion.")
        return
    if choice.isdigit() and 1 <= int(choice) <= len(results):
        _do_promote(root, Path(results[int(choice) - 1]["path"]), results)
        return
    print(f"'{choice}' not recognized; skipped.")


def parse_quality_weights(raw: Optional[str]) -> Optional[Dict[str, float]]:
    """Parse 'spelling=1,punctuation=1,grammar=1,semantics=1' into a weight dict."""
    if not raw:
        return None
    out: Dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid quality weight fragment: {part!r}")
        key, val = part.split("=", 1)
        out[key.strip()] = float(val.strip())
    return out

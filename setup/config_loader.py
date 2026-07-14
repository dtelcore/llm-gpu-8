"""Load training config and resolve dataset corpus without embedding it in JSON."""

from typing import Dict, List

from setup.dataset_setup import DatasetLoader


def resolve_dataset_corpus(dataset_cfg: Dict, data_dir: str = "data") -> List[str]:
    """Return corpus text from inline config or by dataset name/path."""
    corpus = dataset_cfg.get("corpus")
    if corpus:
        return corpus

    name = dataset_cfg.get("name", "minimal")
    loader = DatasetLoader(data_dir=data_dir)
    return loader.load_by_name(name)

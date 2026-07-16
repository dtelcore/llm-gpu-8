"""Load training config and resolve dataset corpus without embedding it in JSON."""

from typing import Dict, List

from paths import DATA_DIR
from setup.dataset_setup import DatasetLoader


def resolve_dataset_corpus(dataset_cfg: Dict, data_dir: str = None) -> List[str]:
    """Return corpus text from inline config or by dataset name/path."""
    corpus = dataset_cfg.get("corpus")
    if corpus:
        return corpus

    name = dataset_cfg.get("name", "minimal")
    loader = DatasetLoader(data_dir=data_dir or str(DATA_DIR))
    return loader.load_by_name(name)

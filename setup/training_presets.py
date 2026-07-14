"""
Scaling presets: bundled model architecture + training hyperparameters.

Used by the --menu wizard in train.py and auto_train.py.
"""

from typing import Dict, Optional, Tuple

from setup.model_config import PRESETS, estimate_vram_footprint

# Bundled presets: model + hyperparameters + default dataset
SCALE_PRESETS: Dict[str, Dict] = {
    'toy': {
        'name': 'Toy Run',
        'tagline': 'Quick smoke test',
        'model_key': 'toy',
        'dataset': 'minimal',
        'hyperparameters': {
            'name': 'Toy Run',
            'learning_rate': 0.01,
            'weight_decay': 0.01,
            'batch_size': 64,
            'num_epochs': 10,
            'warmup_steps': 100,
            'gradient_clip': 1.0,
            'optimizer': 'adamw',
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
        },
    },
    'tiny_stories': {
        'name': 'Tiny Stories (real run)',
        'tagline': 'TinyStories-capable ~1M params',
        'model_key': 'tiny_stories',
        'dataset': 'tiny_stories',
        'hyperparameters': {
            'name': 'Tiny Stories Run',
            'learning_rate': 1e-5,
            'weight_decay': 0.01,
            'batch_size': 32,
            'num_epochs': 1,
            'warmup_steps': 100,
            'gradient_clip': 1.0,
            'optimizer': 'adamw',
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
        },
    },
}


def model_from_preset(preset_key: str, vocab_size: int = 100) -> Dict:
    """Build a model config dict from a PRESETS key."""
    if preset_key not in PRESETS:
        raise ValueError(f"Unknown model preset: {preset_key}")
    cfg = PRESETS[preset_key].copy()
    cfg['vocab_size'] = vocab_size
    return cfg


def apply_scale_preset(scale_key: str, vocab_size: int = 100) -> Tuple[Dict, Dict, str]:
    """Return (model_config, hyperparameters, dataset_name) for a scale preset."""
    if scale_key not in SCALE_PRESETS:
        raise ValueError(f"Unknown scale preset: {scale_key}")
    scale = SCALE_PRESETS[scale_key]
    model = model_from_preset(scale['model_key'], vocab_size=vocab_size)
    hyperparams = scale['hyperparameters'].copy()
    return model, hyperparams, scale['dataset']


def _param_estimate(model_key: str, vocab_size: int = 110) -> int:
    cfg = model_from_preset(model_key, vocab_size=vocab_size)
    return estimate_vram_footprint(cfg)['total_params']


def print_scale_preset_menu(vocab_size: int = 110) -> None:
    """Print the scaling preset table for the training wizard."""
    toy_m = PRESETS['toy']
    ts_m = PRESETS['tiny_stories']
    toy_n = _param_estimate('toy', vocab_size)
    ts_n = _param_estimate('tiny_stories', vocab_size)
    toy_hp = SCALE_PRESETS['toy']['hyperparameters']
    ts_hp = SCALE_PRESETS['tiny_stories']['hyperparameters']

    print("\nScaling presets (model + hyperparameters + recommended dataset):")
    print("-" * 70)
    print(f"  1. Toy Run — {SCALE_PRESETS['toy']['tagline']}")
    print(f"       embed={toy_m['embedding_dim']}  heads={toy_m['num_heads']}  "
          f"layers={toy_m['num_layers']}  seq={toy_m['max_len']}  "
          f"batch={toy_hp['batch_size']}  ~{toy_n:,} params")
    print(f"       Dataset: {SCALE_PRESETS['toy']['dataset']} (built-in)")
    print()
    print(f"  2. Tiny Stories — {SCALE_PRESETS['tiny_stories']['tagline']}")
    print(f"       embed={ts_m['embedding_dim']}  heads={ts_m['num_heads']}  "
          f"layers={ts_m['num_layers']}  seq={ts_m['max_len']}  "
          f"batch={ts_hp['batch_size']}  ~{ts_n:,} params")
    print(f"       Dataset: {SCALE_PRESETS['tiny_stories']['dataset']} (data/*.txt)")
    print()
    print("  3. Custom (pick model + hyperparameters separately)")
    print("-" * 70)


def prompt_scale_preset() -> str:
    """Interactively choose toy, tiny_stories, or custom. Returns preset key."""
    print_scale_preset_menu()
    choice = input("\nSelect scaling preset (1-3) [default=1]: ").strip()
    if choice == '2':
        return 'tiny_stories'
    if choice == '3':
        return 'custom'
    return 'toy'

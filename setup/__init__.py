"""
Setup module for Kepler GT 730 GPU training system.

Provides:
- model_config: Model configuration with presets and interactive builder
- dataset_setup: Dataset selection and loading
- weight_init: Intelligent weight initialization
- training_setup: Complete training setup orchestrator
"""

from setup.model_config import (
    ModelConfigBuilder,
    estimate_vram_footprint,
    load_or_create_config,
    PRESETS as MODEL_PRESETS,
)

from setup.dataset_setup import (
    DatasetLoader,
    DatasetAnalyzer,
    load_dataset_interactive,
    recommend_dataset_for_config,
    BUILTIN_DATASETS,
)

from setup.weight_init import (
    WeightInitializer,
    setup_model_initialization,
    get_init_scales_for_config,
    print_init_scales_table,
)

from setup.training_setup import (
    TrainingSetup,
    quickstart_training_setup,
    HYPERPARAMETER_PRESETS,
)

__all__ = [
    'ModelConfigBuilder',
    'estimate_vram_footprint',
    'load_or_create_config',
    'MODEL_PRESETS',
    'DatasetLoader',
    'DatasetAnalyzer',
    'load_dataset_interactive',
    'recommend_dataset_for_config',
    'BUILTIN_DATASETS',
    'WeightInitializer',
    'setup_model_initialization',
    'get_init_scales_for_config',
    'print_init_scales_table',
    'TrainingSetup',
    'quickstart_training_setup',
    'HYPERPARAMETER_PRESETS',
]

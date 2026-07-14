"""
Complete Model & Training Setup Orchestrator.

Combines:
1. Model configuration (interactive or preset)
2. Dataset selection and loading
3. Weight initialization configuration
4. Training hyperparameters
5. Logging and reproducibility

Single entry point for full training setup.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Running as `python setup/training_setup.py` puts setup/ on sys.path, not the repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from logging_config import logger

from setup.model_config import ModelConfigBuilder, estimate_vram_footprint, load_or_create_config
from setup.dataset_setup import DatasetLoader, DatasetAnalyzer, load_dataset_interactive, recommend_dataset_for_config
from setup.weight_init import setup_model_initialization, get_init_scales_for_config
from setup.training_presets import apply_scale_preset, prompt_scale_preset


# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

HYPERPARAMETER_PRESETS = {
    'conservative': {
        'name': 'Conservative (Stable Training)',
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'batch_size': 2,
        'num_epochs': 5,
        'warmup_steps': 100,
        'gradient_clip': 1.0,
        'optimizer': 'adamw',
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
    },
    'moderate': {
        'name': 'Moderate (Balanced)',
        'learning_rate': 0.01,
        'weight_decay': 0.01,
        'batch_size': 2,
        'num_epochs': 10,
        'warmup_steps': 500,
        'gradient_clip': 1.0,
        'optimizer': 'adamw',
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
    },
    'aggressive': {
        'name': 'Aggressive (Fast Convergence)',
        'learning_rate': 0.1,
        'weight_decay': 0.01,
        'batch_size': 4,
        'num_epochs': 20,
        'warmup_steps': 200,
        'gradient_clip': 0.5,
        'optimizer': 'adamw',
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
    },
}


# ============================================================================
# COMPLETE SETUP ORCHESTRATOR
# ============================================================================

class TrainingSetup:
    """Orchestrates complete training setup."""
    
    def __init__(self, data_dir: str = 'data'):
        """Initialize setup orchestrator.
        
        Args:
            data_dir: Directory auto-scanned for .txt datasets during dataset
                selection (e.g. 'data/tiny_stories.txt' shows up as an option
                alongside the built-in corpora)
        """
        self.model_config = None
        self.dataset = None
        self.dataset_name = None
        self.hyperparams = None
        self.init_scales = None
        self.data_dir = data_dir
        self.scale_preset = None
        self.preset_dataset = None
        self.setup_dir = Path('setup')
        self.config_file = self.setup_dir / 'training_config.json'
        logger.info(f"TrainingSetup initialized (data_dir={data_dir!r})")
    
    def run_interactive_setup(self, use_presets: bool = True) -> Dict:
        """Run complete interactive setup.
        
        Returns:
            Complete training configuration dictionary
        """
        logger.info("Starting interactive training setup...")
        
        print("\n" + "="*70)
        print("COMPLETE TRAINING SETUP WIZARD")
        print("="*70)
        
        # Step 1: Scaling preset (model + hyperparameters) or custom wizard
        print("\n[Step 1/4] MODEL & TRAINING SCALE")
        print("-"*70)
        self.scale_preset = prompt_scale_preset()

        if self.scale_preset in ('toy', 'tiny_stories'):
            self.model_config, self.hyperparams, self.preset_dataset = apply_scale_preset(self.scale_preset)
            builder = ModelConfigBuilder(vocab_size=100)
            builder.config = self.model_config
            builder.print_summary()
            print(f"\n✓ Scale preset: {self.hyperparams['name']}")
            print(f"  Batch size: {self.hyperparams['batch_size']}  |  "
                  f"LR: {self.hyperparams['learning_rate']}  |  "
                  f"Recommended dataset: {self.preset_dataset}")
            logger.info(f"Scale preset applied: {self.scale_preset}")
        else:
            print("\n[Step 1a/4] MODEL CONFIGURATION")
            print("-"*70)
            self._setup_model_config(use_presets)
            print("\n[Step 4/4] TRAINING HYPERPARAMETERS")
            print("-"*70)
            self._setup_hyperparameters(use_presets)
            self.preset_dataset = None
        
        # Step 2: Dataset Selection
        print("\n[Step 2/4] DATASET SELECTION")
        print("-"*70)
        self._setup_dataset()
        
        # Step 3: Weight Initialization
        print("\n[Step 3/4] WEIGHT INITIALIZATION")
        print("-"*70)
        self._setup_weight_initialization()
        
        # Step 4: Hyperparameters (only if not already set by scale preset)
        if self.scale_preset in ('toy', 'tiny_stories'):
            print("\n[Step 4/4] TRAINING HYPERPARAMETERS")
            print("-"*70)
            print(f"✓ Using hyperparameters from '{self.hyperparams['name']}' preset "
                  f"(batch={self.hyperparams['batch_size']}, lr={self.hyperparams['learning_rate']})")
            override = input("Override hyperparameters? (y/n) [default=n]: ").strip().lower()
            if override == 'y':
                self._setup_hyperparameters(use_presets)
        
        # Summary
        self._print_setup_summary()
        
        # Save configuration
        self._save_configuration()
        
        return self.get_complete_config()
    
    def _setup_model_config(self, use_presets: bool = True):
        """Setup model configuration."""
        logger.info("Setting up model configuration...")
        
        builder = ModelConfigBuilder(vocab_size=100)  # Default, will be updated
        
        if use_presets:
            print("\nModel presets:")
            print("  toy / tiny_stories / small / medium / custom")
            choice = input("\nUse preset or custom? [default=toy]: ").strip().lower()
            
            if choice in ['toy', 'tiny_stories', 'tiny', 'small', 'medium']:
                self.model_config = builder.preset_config(choice)
            elif choice == 'custom':
                self.model_config = builder.interactive_config()
            else:
                self.model_config = builder.preset_config('toy')
        else:
            self.model_config = builder.interactive_config()
        
        builder.print_summary()
        logger.info(f"Model config set: {self.model_config['name']}")
    
    def _setup_dataset(self):
        """Setup dataset selection."""
        logger.info("Setting up dataset...")
        
        loader = DatasetLoader(data_dir=self.data_dir)

        if self.preset_dataset:
            recommended = self.preset_dataset
            print(f"\n✓ Preset recommends dataset: {recommended}")
        elif self.model_config:
            recommended = recommend_dataset_for_config(self.model_config)
            print(f"\n✓ Recommended dataset: {recommended}")
        else:
            recommended = None

        if recommended:
            use_recommended = input(
                "Use recommended dataset? (y/n, or 'l' to pick from the list) [default=y]: "
            ).strip().lower()

            if use_recommended == 'l':
                self.dataset, self.dataset_name = load_dataset_interactive(
                    self.model_config, data_dir=self.data_dir,
                )
            elif use_recommended != 'n':
                try:
                    corpus = loader.load_by_name(recommended)
                    self.dataset = corpus
                    self.dataset_name = recommended
                    DatasetAnalyzer(corpus).print_stats()
                except (ValueError, FileNotFoundError) as exc:
                    logger.warning(f"Could not load preset dataset {recommended!r}: {exc}")
                    self.dataset, self.dataset_name = load_dataset_interactive(
                        self.model_config, data_dir=self.data_dir,
                    )
            else:
                self.dataset, self.dataset_name = load_dataset_interactive(
                    self.model_config, data_dir=self.data_dir,
                )
        else:
            self.dataset, self.dataset_name = load_dataset_interactive(
                self.model_config, data_dir=self.data_dir,
            )
        
        # Update vocab size in model config
        if self.dataset:
            vocab_size = len(set(' '.join(self.dataset)))
            self.model_config['vocab_size'] = vocab_size
            logger.info(f"Updated vocab_size from corpus: {vocab_size}")
    
    def _setup_weight_initialization(self):
        """Setup weight initialization."""
        logger.info("Setting up weight initialization...")
        
        if self.model_config:
            self.init_scales = setup_model_initialization(self.model_config)
    
    def _setup_hyperparameters(self, use_presets: bool = True):
        """Setup training hyperparameters."""
        logger.info("Setting up hyperparameters...")
        
        if use_presets:
            print("\nHyperparameter presets:")
            print("  1. Conservative (stable, slow)")
            print("  2. Moderate (balanced)")
            print("  3. Aggressive (fast, unstable)")
            print("  4. Custom")
            
            choice = input("\nSelect (1-4) [default=2]: ").strip()
            
            if choice == '1':
                self.hyperparams = HYPERPARAMETER_PRESETS['conservative'].copy()
            elif choice == '3':
                self.hyperparams = HYPERPARAMETER_PRESETS['aggressive'].copy()
            elif choice == '4':
                self.hyperparams = self._build_custom_hyperparams()
            else:
                self.hyperparams = HYPERPARAMETER_PRESETS['moderate'].copy()
        else:
            self.hyperparams = self._build_custom_hyperparams()
        
        logger.info(f"Hyperparameters set: {self.hyperparams['name']}")
    
    def _build_custom_hyperparams(self) -> Dict:
        """Build custom hyperparameters."""
        print("\n[Custom Hyperparameters]")
        
        hyperparams = {
            'name': 'Custom Configuration',
            'learning_rate': float(input("Learning rate [default=0.01]: ") or "0.01"),
            'weight_decay': float(input("Weight decay [default=0.01]: ") or "0.01"),
            'batch_size': int(input("Batch size [default=2]: ") or "2"),
            'num_epochs': int(input("Number of epochs [default=10]: ") or "10"),
            'warmup_steps': int(input("Warmup steps [default=500]: ") or "500"),
            'gradient_clip': float(input("Gradient clip norm [default=1.0]: ") or "1.0"),
            'optimizer': 'adamw',
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
        }
        
        return hyperparams
    
    def _print_setup_summary(self):
        """Print comprehensive setup summary."""
        print("\n" + "="*70)
        print("TRAINING SETUP SUMMARY")
        print("="*70)
        
        print("\n[MODEL]")
        print(f"  Name:                  {self.model_config.get('name', 'Custom')}")
        print(f"  Vocabulary Size:       {self.model_config['vocab_size']}")
        print(f"  Embedding Dimension:   {self.model_config['embedding_dim']}")
        print(f"  Attention Heads:       {self.model_config['num_heads']}")
        print(f"  Transformer Layers:    {self.model_config['num_layers']}")
        print(f"  Max Sequence Length:   {self.model_config['max_len']}")
        
        estimate = estimate_vram_footprint(self.model_config)
        print(f"  Parameter Count:       {estimate['total_params']:,}")
        print(f"  Estimated VRAM:        {estimate['total_mb']:.2f} MB")
        
        print("\n[DATASET]")
        print(f"  Name:                  {self.dataset_name}")
        print(f"  Number of Sentences:   {len(self.dataset)}")
        
        analyzer = DatasetAnalyzer(self.dataset)
        print(f"  Total Characters:      {analyzer.stats['num_characters']:,}")
        print(f"  Unique Characters:     {analyzer.stats['unique_characters']}")
        print(f"  Avg Sentence Length:   {analyzer.stats['avg_sentence_length']:.2f} words")
        
        print("\n[TRAINING HYPERPARAMETERS]")
        print(f"  Preset:                {self.hyperparams['name']}")
        print(f"  Learning Rate:         {self.hyperparams['learning_rate']}")
        print(f"  Weight Decay:          {self.hyperparams['weight_decay']}")
        print(f"  Batch Size:            {self.hyperparams['batch_size']}")
        print(f"  Number of Epochs:      {self.hyperparams['num_epochs']}")
        print(f"  Warmup Steps:          {self.hyperparams['warmup_steps']}")
        print(f"  Gradient Clip Norm:    {self.hyperparams['gradient_clip']}")
        print(f"  Optimizer:             {self.hyperparams['optimizer']}")
        
        print("\n" + "="*70)
    
    def _save_configuration(self):
        """Save complete configuration to file."""
        # In-memory corpus for the current session (not written to training_config.json).
        config = self.get_complete_config()
        config['dataset']['corpus'] = self.dataset
        
        # Ensure setup directory exists
        self.setup_dir.mkdir(exist_ok=True)
        
        # Save to JSON
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {self.config_file}")
        print(f"\n✓ Configuration saved to {self.config_file}")
    
    def get_complete_config(self) -> Dict:
        """Get complete training configuration.
        
        Returns:
            Dict with all settings
        """
        return {
            'model': self.model_config,
            'dataset': {
                'name': self.dataset_name,
                'vocab_size': self.model_config['vocab_size'],
                'num_sentences': len(self.dataset) if self.dataset else 0,
            },
            'weight_initialization': self.init_scales,
            'hyperparameters': self.hyperparams,
            'metadata': {
                'created': str(np.datetime64('today')),
            }
        }
    
    def load_configuration(self, filepath: str) -> Dict:
        """Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Complete configuration dictionary
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.model_config = config['model']
        self.dataset = config['dataset'].get('corpus', [])
        self.dataset_name = config['dataset'].get('name', 'unknown')
        self.init_scales = config.get('weight_initialization', {})
        self.hyperparams = config.get('hyperparameters', {})
        
        logger.info(f"Configuration loaded from {filepath}")
        return config


# ============================================================================
# QUICK START FUNCTION
# ============================================================================

def quickstart_training_setup(interactive: bool = True, data_dir: str = 'data') -> Dict:
    """Quick start training setup.
    
    Args:
        interactive: If True, run interactive wizard; else use all defaults
        data_dir: Directory auto-scanned for .txt datasets during dataset
            selection (only relevant when interactive=True)
        
    Returns:
        Complete training configuration
    """
    setup = TrainingSetup(data_dir=data_dir)
    
    if interactive:
        return setup.run_interactive_setup()
    else:
        # Use all defaults
        setup.model_config = {'vocab_size': 92, 'max_len': 8, 'embedding_dim': 16,
                             'num_heads': 2, 'num_layers': 1, 'dropout_prob': 0.0,
                             'init_scale': 0.02, 'name': 'Toy Run'}
        setup.dataset = ["cuda training operational", "gpu acceleration optimizes"]
        setup.dataset_name = 'minimal'
        setup.init_scales = get_init_scales_for_config(setup.model_config)
        setup.hyperparams = apply_scale_preset('toy')[1]
        
        return setup.get_complete_config()


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(description="Interactive training setup wizard")
    _parser.add_argument("--data-dir", type=str, default="data", help="Directory auto-scanned for .txt datasets (default: data)")
    _args = _parser.parse_args()

    config = quickstart_training_setup(interactive=True, data_dir=_args.data_dir)
    print("\nSetup complete. Keys:", ", ".join(config.keys()))

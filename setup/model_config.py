"""
Interactive Model Configuration Setup for Kepler GT 730 GPU Training.

Provides:
1. Predefined configurations (Tiny, Small, Medium) optimized for VRAM budgets
2. Interactive configuration builder with validation
3. Smart weight initialization based on layer type and fan-in/fan-out
4. Configuration persistence (save/load from JSON)
5. Memory footprint estimation

Target: NVIDIA GeForce GT 730 (1-2GB VRAM constraint)
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from logging_config import logger


# ============================================================================
# PREDEFINED MODEL CONFIGURATIONS
# ============================================================================

PRESETS = {
    'tiny': {
        'name': 'Tiny (Testing)',
        'vocab_size': None,  # Inferred from tokenizer
        'max_len': 8,
        'embedding_dim': 16,
        'num_heads': 2,
        'num_layers': 1,
        'dropout_prob': 0.0,
        'init_scale': 0.02,
        'description': 'Minimal config for quick testing. ~50KB parameters. Batch=2, SeqLen=8.',
    },
    'small': {
        'name': 'Small (Development)',
        'vocab_size': None,
        'max_len': 16,
        'embedding_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'dropout_prob': 0.1,
        'init_scale': 0.02,
        'description': 'Small model for development. ~200KB parameters. Batch=2, SeqLen=16.',
    },
    'medium': {
        'name': 'Medium (Production)',
        'vocab_size': None,
        'max_len': 32,
        'embedding_dim': 128,
        'num_heads': 8,
        'num_layers': 3,
        'dropout_prob': 0.1,
        'init_scale': 0.02,
        'description': 'Medium model for production training. ~1MB parameters. Batch=2, SeqLen=32.',
    },
}


# ============================================================================
# WEIGHT INITIALIZATION UTILITIES
# ============================================================================

def calculate_init_scale(layer_type: str, fan_in: int, fan_out: Optional[int] = None) -> float:
    """Calculate appropriate weight initialization scale based on layer type.
    
    Args:
        layer_type: 'embedding', 'attention', 'mlp_expand', 'mlp_contract', 'output'
        fan_in: Input dimension (rows in weight matrix)
        fan_out: Output dimension (columns in weight matrix), optional for some types
        
    Returns:
        init_scale: Standard deviation for normal distribution initialization
    """
    if layer_type == 'embedding':
        # Embeddings benefit from smaller scale to avoid extreme values
        return 1.0 / np.sqrt(fan_in)
    
    elif layer_type == 'attention':
        # Attention projections: scaled by input dimension
        return 1.0 / np.sqrt(fan_in)
    
    elif layer_type == 'mlp_expand':
        # First MLP layer: expand from C → 4C
        return 1.0 / np.sqrt(fan_in)
    
    elif layer_type == 'mlp_contract':
        # Second MLP layer: contract from 4C → C
        return 1.0 / np.sqrt(fan_in)
    
    elif layer_type == 'output':
        # Output projection (lm_head): critical for stability
        return 1.0 / np.sqrt(fan_in)
    
    else:
        # Default: He initialization
        return np.sqrt(2.0 / fan_in)


def estimate_vram_footprint(config_dict: Dict) -> Dict[str, float]:
    """Estimate GPU memory footprint for given configuration.
    
    Args:
        config_dict: Configuration dictionary with model parameters
        
    Returns:
        Dict with breakdown of parameter sizes:
        - total_params: Total number of float32 parameters
        - total_bytes: Total bytes (assuming float32 = 4 bytes)
        - total_mb: Total megabytes
        - breakdown: Per-layer component sizes
    """
    vocab_size = config_dict.get('vocab_size', 100)
    max_len = config_dict['max_len']
    C = config_dict['embedding_dim']
    H = config_dict['num_heads']
    L = config_dict['num_layers']
    
    total_params = 0
    breakdown = {}
    
    # 1. Token embeddings [vocab_size, C]
    token_embed = vocab_size * C
    total_params += token_embed
    breakdown['token_embedding'] = token_embed
    
    # 2. Position embeddings [max_len, C]
    pos_embed = max_len * C
    total_params += pos_embed
    breakdown['position_embedding'] = pos_embed
    
    # 3. Per transformer block:
    # - LayerNorm: 2 × (gamma [C] + beta [C]) = 4C per block
    # - Attention: QKV proj [C, 3C] + output proj [C, C] = 4C² params
    # - MLP: expand [C, 4C] + contract [4C, C] = 8C² params
    # - Biases: 3C (QKV) + C (output) + 4C (expand) + C (contract) = 9C
    
    per_block_params = (
        4 * C +  # Layer norms (2 gammas + 2 betas)
        C * (3*C) +  # QKV projection
        C * C +  # Output projection
        C * (4*C) +  # MLP expand
        (4*C) * C +  # MLP contract
        (3*C + C + 4*C + C)  # All biases
    )
    
    transformer_params = L * per_block_params
    total_params += transformer_params
    breakdown['transformer_blocks'] = transformer_params
    
    # 4. Final layer norm [C] + [C]
    final_ln = 2 * C
    total_params += final_ln
    breakdown['final_layernorm'] = final_ln
    
    # 5. Output projection [C, vocab_size]
    output_proj = C * vocab_size
    total_params += output_proj
    breakdown['output_projection'] = output_proj
    
    # Convert to bytes (float32 = 4 bytes per param)
    total_bytes = total_params * 4
    total_mb = total_bytes / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'total_bytes': total_bytes,
        'total_mb': total_mb,
        'breakdown': {k: v for k, v in breakdown.items()},
    }


# ============================================================================
# CONFIGURATION BUILDER
# ============================================================================

class ModelConfigBuilder:
    """Interactive builder for model configuration with validation."""
    
    def __init__(self, vocab_size: int):
        """Initialize builder with tokenizer vocabulary size.
        
        Args:
            vocab_size: Number of tokens in vocabulary
        """
        self.vocab_size = vocab_size
        self.config = None
        logger.info(f"ModelConfigBuilder initialized with vocab_size={vocab_size}")
    
    def preset_config(self, preset_name: str) -> Dict:
        """Load a predefined configuration preset.
        
        Args:
            preset_name: One of 'tiny', 'small', 'medium'
            
        Returns:
            Configuration dictionary
        """
        if preset_name not in PRESETS:
            logger.error(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
            raise ValueError(f"Unknown preset: {preset_name}")
        
        config = PRESETS[preset_name].copy()
        config['vocab_size'] = self.vocab_size
        
        self.config = config
        logger.info(f"Loaded preset: {preset_name} ({config['name']})")
        return config
    
    def interactive_config(self) -> Dict:
        """Build configuration interactively with user input.
        
        Returns:
            Validated configuration dictionary
        """
        logger.info("Starting interactive configuration builder...")
        print("\n" + "="*70)
        print("INTERACTIVE MODEL CONFIGURATION BUILDER")
        print("="*70)
        
        print("\n[Quick Start] Choose a preset or customize:")
        print("  1. Tiny (minimal testing)")
        print("  2. Small (development)")
        print("  3. Medium (production)")
        print("  4. Custom (full customization)")
        
        choice = input("\nSelect (1-4): ").strip()
        
        if choice in ['1', '2', '3']:
            preset_map = {'1': 'tiny', '2': 'small', '3': 'medium'}
            return self.preset_config(preset_map[choice])
        
        elif choice == '4':
            return self._build_custom_config()
        
        else:
            logger.warning(f"Invalid choice: {choice}, defaulting to tiny")
            return self.preset_config('tiny')
    
    def _build_custom_config(self) -> Dict:
        """Build custom configuration step-by-step."""
        print("\n[Custom Configuration]")
        
        config = {
            'vocab_size': self.vocab_size,
            'name': 'Custom Configuration',
        }
        
        # Sequence length
        while True:
            try:
                max_len = int(input("Max sequence length (8-256) [default=16]: ") or "16")
                if 8 <= max_len <= 256:
                    config['max_len'] = max_len
                    break
                print("  ⚠ Must be between 8 and 256")
            except ValueError:
                print("  ⚠ Invalid input, using default 16")
                config['max_len'] = 16
                break
        
        # Embedding dimension (must be divisible by num_heads)
        while True:
            try:
                embedding_dim = int(input("Embedding dimension (16-512) [default=64]: ") or "64")
                if 16 <= embedding_dim <= 512 and embedding_dim % 8 == 0:
                    config['embedding_dim'] = embedding_dim
                    break
                print("  ⚠ Must be between 16-512 and divisible by 8")
            except ValueError:
                print("  ⚠ Invalid input, using default 64")
                config['embedding_dim'] = 64
                break
        
        # Number of heads
        max_heads = config['embedding_dim'] // 8
        while True:
            try:
                num_heads = int(input(f"Number of attention heads (2-{max_heads}) [default=4]: ") or "4")
                if 2 <= num_heads <= max_heads and config['embedding_dim'] % num_heads == 0:
                    config['num_heads'] = num_heads
                    break
                print(f"  ⚠ Must be between 2-{max_heads} and divide {config['embedding_dim']} evenly")
            except ValueError:
                print("  ⚠ Invalid input, using default 4")
                config['num_heads'] = 4
                break
        
        # Number of layers
        while True:
            try:
                num_layers = int(input("Number of transformer layers (1-8) [default=2]: ") or "2")
                if 1 <= num_layers <= 8:
                    config['num_layers'] = num_layers
                    break
                print("  ⚠ Must be between 1 and 8")
            except ValueError:
                print("  ⚠ Invalid input, using default 2")
                config['num_layers'] = 2
                break
        
        # Dropout probability
        while True:
            try:
                dropout = float(input("Dropout probability (0.0-0.5) [default=0.1]: ") or "0.1")
                if 0.0 <= dropout <= 0.5:
                    config['dropout_prob'] = dropout
                    break
                print("  ⚠ Must be between 0.0 and 0.5")
            except ValueError:
                print("  ⚠ Invalid input, using default 0.1")
                config['dropout_prob'] = 0.1
                break
        
        # Validate configuration
        self._validate_config(config)
        self.config = config
        logger.info(f"Custom configuration created: {config}")
        return config
    
    def _validate_config(self, config: Dict) -> bool:
        """Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, raises exception otherwise
        """
        errors = []
        
        # Vocab size check
        if config.get('vocab_size', 0) < 2:
            errors.append("vocab_size must be >= 2")
        
        # Embedding dim must be divisible by num_heads
        if config['embedding_dim'] % config['num_heads'] != 0:
            errors.append(f"embedding_dim ({config['embedding_dim']}) must be divisible by num_heads ({config['num_heads']})")
        
        # Max length sanity
        if not (1 <= config['max_len'] <= 1024):
            errors.append("max_len must be between 1 and 1024")
        
        # Num layers sanity
        if not (1 <= config['num_layers'] <= 32):
            errors.append("num_layers must be between 1 and 32")
        
        # Dropout range
        if not (0.0 <= config['dropout_prob'] <= 1.0):
            errors.append("dropout_prob must be between 0.0 and 1.0")
        
        if errors:
            error_msg = "\n".join(f"  ✗ {e}" for e in errors)
            logger.error(f"Configuration validation failed:\n{error_msg}")
            raise ValueError(f"Invalid configuration:\n{error_msg}")
        
        logger.info("✓ Configuration validation passed")
        return True
    
    def estimate_memory(self) -> Dict:
        """Estimate VRAM footprint of current configuration.
        
        Returns:
            Memory estimation dictionary
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded. Call preset_config() or interactive_config() first.")
        
        estimate = estimate_vram_footprint(self.config)
        logger.info(f"Memory estimate: {estimate['total_mb']:.2f} MB")
        return estimate
    
    def save_config(self, filepath: str) -> str:
        """Save configuration to JSON file.
        
        Args:
            filepath: Path to save JSON config
            
        Returns:
            Absolute path to saved file
        """
        if self.config is None:
            raise RuntimeError("No configuration to save. Build one first.")
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        config_with_metadata = self.config.copy()
        config_with_metadata['_created'] = str(np.datetime64('today'))
        config_with_metadata['_memory_estimate_mb'] = estimate_vram_footprint(self.config)['total_mb']
        
        with open(filepath, 'w') as f:
            json.dump(config_with_metadata, f, indent=2)
        
        abs_path = os.path.abspath(filepath)
        logger.info(f"Configuration saved to {abs_path}")
        return abs_path
    
    def load_config(self, filepath: str) -> Dict:
        """Load configuration from JSON file.
        
        Args:
            filepath: Path to load JSON config from
            
        Returns:
            Loaded configuration dictionary
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Remove metadata fields
        config = {k: v for k, v in config.items() if not k.startswith('_')}
        
        # Validate
        self._validate_config(config)
        self.config = config
        logger.info(f"Configuration loaded from {filepath}")
        return config
    
    def get_config(self) -> Dict:
        """Get current configuration.
        
        Returns:
            Current configuration dictionary
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        return self.config.copy()
    
    def print_summary(self):
        """Print configuration summary with memory estimates."""
        if self.config is None:
            logger.warning("No configuration to display")
            return
        
        estimate = estimate_vram_footprint(self.config)
        
        print("\n" + "="*70)
        print("MODEL CONFIGURATION SUMMARY")
        print("="*70)
        print(f"Name:                  {self.config.get('name', 'Custom')}")
        print(f"Vocabulary Size:       {self.config['vocab_size']}")
        print(f"Max Sequence Length:   {self.config['max_len']}")
        print(f"Embedding Dimension:   {self.config['embedding_dim']}")
        print(f"  → Head Dimension:    {self.config['embedding_dim'] // self.config['num_heads']}")
        print(f"Attention Heads:       {self.config['num_heads']}")
        print(f"Transformer Layers:    {self.config['num_layers']}")
        print(f"Dropout Probability:   {self.config['dropout_prob']:.2f}")
        print()
        print("PARAMETER COUNTS")
        print("-" * 70)
        print(f"Total Parameters:      {estimate['total_params']:,}")
        print(f"Total Size (float32):  {estimate['total_mb']:.2f} MB")
        print()
        print("BREAKDOWN BY COMPONENT")
        print("-" * 70)
        for component, count in estimate['breakdown'].items():
            pct = 100 * count / estimate['total_params']
            print(f"  {component:30s}: {count:10,} params ({pct:5.1f}%)")
        print("="*70 + "\n")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_or_create_config(vocab_size: int, config_path: Optional[str] = None, 
                          interactive: bool = False) -> Dict:
    """Load existing config or create new one interactively.
    
    Args:
        vocab_size: Number of tokens in vocabulary
        config_path: Optional path to load config from
        interactive: If True, always prompt user interactively
        
    Returns:
        Configuration dictionary
    """
    builder = ModelConfigBuilder(vocab_size)
    
    # Try to load existing config
    if config_path and os.path.exists(config_path) and not interactive:
        logger.info(f"Loading configuration from {config_path}...")
        return builder.load_config(config_path)
    
    # Build interactively
    if interactive:
        config = builder.interactive_config()
    else:
        # Default to tiny
        config = builder.preset_config('tiny')
    
    # Save config if path provided
    if config_path:
        builder.save_config(config_path)
    
    return config

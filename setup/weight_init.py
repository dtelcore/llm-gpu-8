"""
Intelligent Weight Initialization Utilities for GPT Model.

Provides proper initialization strategies based on:
1. Layer type (embedding, attention, mlp, output projection)
2. Fan-in/fan-out dimensions
3. Model configuration
4. Best practices from GPT/Transformer literature

Ensures training stability and prevents gradient explosion/vanishing.

Reference: "Understanding the difficulty of training deep feedforward neural networks"
           Xavier et al., AISTATS 2010
"""

import numpy as np
from typing import Dict, Tuple
from logging_config import logger


# ============================================================================
# INITIALIZATION SCALES & SCHEDULES
# ============================================================================

class WeightInitializer:
    """Intelligent weight initialization based on layer type and config."""
    
    @staticmethod
    def layer_init_scale(layer_type: str, fan_in: int, fan_out: int, 
                        depth: int = 1, total_layers: int = 1) -> float:
        """Calculate initialization scale for a specific layer.
        
        Args:
            layer_type: Type of layer ('embedding', 'qkv_proj', 'output_proj',
                                       'mlp_expand', 'mlp_contract', 'lm_head')
            fan_in: Input dimension (number of input features)
            fan_out: Output dimension (number of output features)
            depth: Layer position in network (1-indexed, affects scale for depth-dependent init)
            total_layers: Total number of layers in network
            
        Returns:
            Standard deviation for normal distribution initialization
        """
        
        # Embedding layers: small variance to avoid extreme values
        if layer_type == 'embedding':
            # Use smaller scale for embeddings: std = 1/sqrt(fan_in)
            scale = 1.0 / np.sqrt(fan_in)
            logger.debug(f"Embedding init scale: {scale:.6f} (fan_in={fan_in})")
            return scale
        
        # Attention projection layers
        elif layer_type == 'qkv_proj':
            # Query/Key/Value projections: std = 1/sqrt(fan_in)
            # These are the first projections, use conservative scale
            scale = 1.0 / np.sqrt(fan_in)
            logger.debug(f"QKV projection init scale: {scale:.6f} (fan_in={fan_in})")
            return scale
        
        elif layer_type == 'output_proj':
            # Output projections: 1/sqrt(fan_in) per transformer literature
            scale = 1.0 / np.sqrt(fan_in)
            logger.debug(f"Output projection init scale: {scale:.6f} (fan_in={fan_in})")
            return scale
        
        # Feed-forward network layers
        elif layer_type == 'mlp_expand':
            # Expanding layer (C -> 4C): std = 1/sqrt(fan_in)
            scale = 1.0 / np.sqrt(fan_in)
            logger.debug(f"MLP expand init scale: {scale:.6f} (fan_in={fan_in})")
            return scale
        
        elif layer_type == 'mlp_contract':
            # Contracting layer (4C -> C): std = 1/sqrt(fan_in)
            # This is critical for gradient flow
            scale = 1.0 / np.sqrt(fan_in)
            logger.debug(f"MLP contract init scale: {scale:.6f} (fan_in={fan_in})")
            return scale
        
        # Language model head (final projection to vocabulary)
        elif layer_type == 'lm_head':
            # Final projection: use standard fan-in scaling
            scale = 1.0 / np.sqrt(fan_in)
            logger.debug(f"LM head init scale: {scale:.6f} (fan_in={fan_in})")
            return scale
        
        # Layer normalization parameters
        elif layer_type == 'layernorm':
            # Gamma (scale) initialized to 1.0, beta (offset) to 0.0
            # These are set separately, not here
            logger.debug(f"LayerNorm parameters: gamma=1.0, beta=0.0")
            return 1.0
        
        # Default: He initialization for ReLU networks
        else:
            scale = np.sqrt(2.0 / fan_in)  # He initialization
            logger.warning(f"Unknown layer type '{layer_type}', using He init: {scale:.6f}")
            return scale
    
    @staticmethod
    def initialize_weights(weights: np.ndarray, layer_type: str, fan_in: int, 
                          fan_out: int, depth: int = 1, total_layers: int = 1) -> np.ndarray:
        """Initialize weight matrix with proper scale.
        
        Args:
            weights: Weight matrix to initialize (float32)
            layer_type: Type of layer (see layer_init_scale)
            fan_in: Input dimension
            fan_out: Output dimension
            depth: Layer position (1-indexed)
            total_layers: Total number of layers
            
        Returns:
            Initialized weight matrix
        """
        scale = WeightInitializer.layer_init_scale(
            layer_type, fan_in, fan_out, depth, total_layers
        )
        
        # Sample from normal distribution N(0, scale^2)
        weights = np.random.normal(0.0, scale, size=weights.shape).astype(np.float32)
        logger.debug(f"Initialized {layer_type} weights: shape={weights.shape}, scale={scale:.6f}")
        
        return weights
    
    @staticmethod
    def bias_init(bias_shape: Tuple[int], layer_type: str = 'default') -> np.ndarray:
        """Initialize bias vector.
        
        Args:
            bias_shape: Shape of bias vector
            layer_type: Type of layer (for special cases)
            
        Returns:
            Initialized bias vector (zeros)
        """
        # Standard practice: initialize all biases to zero
        bias = np.zeros(bias_shape, dtype=np.float32)
        logger.debug(f"Initialized bias: shape={bias_shape}")
        return bias
    
    @staticmethod
    def layernorm_init(shape: Tuple[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize LayerNorm parameters.
        
        Args:
            shape: Shape of gamma (and beta)
            
        Returns:
            Tuple of (gamma, beta) both float32
        """
        gamma = np.ones(shape, dtype=np.float32)  # Scale factor (multiplicative)
        beta = np.zeros(shape, dtype=np.float32)  # Offset (additive)
        logger.debug(f"Initialized LayerNorm: shape={shape}")
        return gamma, beta


# ============================================================================
# CONFIGURATION-BASED INITIALIZATION
# ============================================================================

def get_init_scales_for_config(config: Dict) -> Dict[str, float]:
    """Get all initialization scales needed for a model configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Dict mapping layer types to their initialization scales
    """
    C = config['embedding_dim']
    vocab_size = config['vocab_size']
    max_len = config['max_len']
    H = config['num_heads']
    L = config['num_layers']
    
    head_dim = C // H
    
    scales = {}
    
    # Token embedding [vocab_size, C]
    scales['token_embedding'] = WeightInitializer.layer_init_scale(
        'embedding', vocab_size, C
    )
    
    # Position embedding [max_len, C]
    scales['position_embedding'] = WeightInitializer.layer_init_scale(
        'embedding', max_len, C
    )
    
    # Per-layer scales (same for all transformer blocks due to shared initialization)
    
    # QKV projection [C, 3C]
    scales['qkv_proj'] = WeightInitializer.layer_init_scale(
        'qkv_proj', C, 3*C
    )
    
    # Output projection (attention) [C, C]
    scales['attention_output_proj'] = WeightInitializer.layer_init_scale(
        'output_proj', C, C
    )
    
    # MLP expand [C, 4C]
    scales['mlp_expand'] = WeightInitializer.layer_init_scale(
        'mlp_expand', C, 4*C
    )
    
    # MLP contract [4C, C]
    scales['mlp_contract'] = WeightInitializer.layer_init_scale(
        'mlp_contract', 4*C, C
    )
    
    # Output projection (lm_head) [C, vocab_size]
    scales['lm_head'] = WeightInitializer.layer_init_scale(
        'lm_head', C, vocab_size
    )
    
    logger.info(f"Generated initialization scales for config")
    return scales


def print_init_scales_table(config: Dict):
    """Print a table of initialization scales for debugging.
    
    Args:
        config: Model configuration dictionary
    """
    scales = get_init_scales_for_config(config)
    
    print("\nINITIALIZATION SCALES TABLE")
    print("-" * 70)
    print(f"{'Layer Type':<30s} {'Scale':>12s}")
    print("-" * 70)
    
    for layer_type, scale in sorted(scales.items()):
        print(f"{layer_type:<30s} {scale:>12.8f}")
    
    print("-" * 70 + "\n")


# ============================================================================
# VALIDATION & DIAGNOSTICS
# ============================================================================

def validate_weight_distribution(weights: np.ndarray, expected_scale: float, 
                                tolerance: float = 0.2) -> bool:
    """Validate that initialized weights match expected scale.
    
    Args:
        weights: Weight matrix to validate
        expected_scale: Expected standard deviation
        tolerance: Acceptable deviation from expected (fraction, 0.2 = ±20%)
        
    Returns:
        True if validation passes
    """
    actual_std = np.std(weights)
    actual_mean = np.mean(weights)
    
    # Check mean is close to zero
    if abs(actual_mean) > 0.1 * expected_scale:
        logger.warning(f"Weight mean={actual_mean:.6f} deviates from 0.0")
    
    # Check std is close to expected
    tolerance_range = (expected_scale * (1 - tolerance), expected_scale * (1 + tolerance))
    is_valid = tolerance_range[0] <= actual_std <= tolerance_range[1]
    
    if not is_valid:
        logger.warning(f"Weight std={actual_std:.6f} outside range {tolerance_range}")
    
    return is_valid


def print_weight_stats(weights: np.ndarray, layer_name: str):
    """Print statistics about weight initialization.
    
    Args:
        weights: Weight matrix
        layer_name: Name of layer for logging
    """
    print(f"\nWeight Statistics: {layer_name}")
    print(f"  Shape:       {weights.shape}")
    print(f"  Mean:        {np.mean(weights):.8f}")
    print(f"  Std Dev:     {np.std(weights):.8f}")
    print(f"  Min:         {np.min(weights):.8f}")
    print(f"  Max:         {np.max(weights):.8f}")
    print(f"  Range:       [{np.min(weights):.4f}, {np.max(weights):.4f}]")


# ============================================================================
# QUICK START
# ============================================================================

def setup_model_initialization(config: Dict, seed: int = 42) -> Dict[str, float]:
    """Setup and validate model initialization for a configuration.
    
    Args:
        config: Model configuration dictionary
        seed: Random seed for reproducibility
        
    Returns:
        Dict mapping layer types to initialization scales
    """
    np.random.seed(seed)
    
    logger.info(f"Setting up weight initialization with seed={seed}")
    
    # Get initialization scales
    scales = get_init_scales_for_config(config)
    
    # Print diagnostic table
    print_init_scales_table(config)
    
    # Log for debugging
    logger.info(f"Weight initialization configured:")
    for layer_type, scale in scales.items():
        logger.debug(f"  {layer_type}: {scale:.8f}")
    
    return scales

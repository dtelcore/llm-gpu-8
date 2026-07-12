# Setup Directory: Complete Training Configuration System

Current state: setup still drives model, dataset, and weight-init configuration, but the recommended validation path now includes the shared probe checks after each fresh checkpoint save.

Complete infrastructure for model configuration, dataset selection, weight initialization, and training hyperparameter setup for the Kepler GT 730 GPU training system.

---

## 📋 Overview

The setup system provides:

1. **Model Configuration** (`model_config.py`)
   - Predefined configurations (Tiny, Small, Medium)
   - Interactive configuration builder
   - VRAM footprint estimation
   - Configuration persistence (save/load JSON)

2. **Dataset Management** (`dataset_setup.py`)
   - Built-in tiny datasets for testing
   - File-based dataset loading
   - Dataset discovery from `data/` directory
   - Dataset recommendations based on model config
   - Corpus statistics and analysis

3. **Weight Initialization** (`weight_init.py`)
   - Intelligent initialization based on layer type
   - Fan-in/fan-out scaling (He, Xavier initialization)
   - Configuration-aware initialization scales
   - Weight distribution validation

4. **Training Setup Orchestrator** (`training_setup.py`)
   - Complete interactive setup wizard
   - Hyperparameter presets (Conservative, Moderate, Aggressive)
   - Configuration verification
   - Unified configuration saving/loading

---

## 🚀 Quick Start

### Option 1: Interactive Setup (Recommended)

```python
from setup import quickstart_training_setup

# Run interactive wizard
config = quickstart_training_setup(interactive=True)

# config contains:
# - model: Model configuration (vocab_size, embedding_dim, etc.)
# - dataset: Dataset with corpus and metadata
# - weight_initialization: Initialization scales for each layer type
# - hyperparameters: Training settings (learning_rate, batch_size, etc.)
```

### Option 2: Programmatic Setup

```python
from setup import ModelConfigBuilder, DatasetLoader, WeightInitializer, HYPERPARAMETER_PRESETS

# Create model config
builder = ModelConfigBuilder(vocab_size=92)
model_config = builder.preset_config('tiny')

# Load dataset
loader = DatasetLoader()
corpus = loader.load_builtin('minimal')

# Setup weight initialization
init_scales = setup_model_initialization(model_config)

# Use hyperparameters
hyperparams = HYPERPARAMETER_PRESETS['moderate']
```

### Option 3: Use Defaults (No Interaction)

```python
from setup import quickstart_training_setup

# Uses all defaults without prompting
config = quickstart_training_setup(interactive=False)
```

---

## 📁 File Structure

```
setup/
├── __init__.py                 # Module exports
├── model_config.py            # Model configuration system
├── dataset_setup.py           # Dataset loading and management
├── weight_init.py             # Weight initialization utilities
├── training_setup.py          # Complete setup orchestrator
├── training_config.json       # Saved configuration (generated)
└── README.md                  # This file
```

---

## 🎛️ Component Details

### 1. Model Configuration (`model_config.py`)

**Predefined Presets:**

| Preset | Vocab | Embedding | Heads | Layers | Max Seq | Params | VRAM |
|--------|-------|-----------|-------|--------|---------|--------|------|
| **Tiny** | Auto | 16 | 2 | 1 | 8 | ~20K | ~80KB |
| **Small** | Auto | 64 | 4 | 2 | 16 | ~200K | ~800KB |
| **Medium** | Auto | 128 | 8 | 3 | 32 | ~1M | ~4MB |

**Usage:**

```python
from setup import ModelConfigBuilder

builder = ModelConfigBuilder(vocab_size=92)

# Load preset
config = builder.preset_config('tiny')

# Or build interactively
config = builder.interactive_config()

# Estimate VRAM
estimate = builder.estimate_memory()
print(f"Total parameters: {estimate['total_params']:,}")
print(f"VRAM required: {estimate['total_mb']:.2f} MB")

# Save and load
builder.save_config('setup/training_config.json')
builder.load_config('setup/training_config.json')

# Print summary
builder.print_summary()
```

**Configuration Dictionary:**

```python
{
    'name': 'Tiny (Testing)',
    'vocab_size': 92,          # From tokenizer
    'max_len': 8,              # Sequence length
    'embedding_dim': 16,       # Hidden dimension
    'num_heads': 2,            # Attention heads
    'num_layers': 1,           # Transformer blocks
    'dropout_prob': 0.0,       # Dropout rate
    'head_dim': 8,             # Auto-computed
}
```

### 2. Dataset Management (`dataset_setup.py`)

**Built-in Datasets:**

| Name | Sentences | Use Case |
|------|-----------|----------|
| `minimal` | 3 | Quick unit testing |
| `tiny_code` | 5 | Code generation experiments |
| `tiny_english` | 5 | Language modeling |

**Usage:**

```python
from setup import DatasetLoader, DatasetAnalyzer

loader = DatasetLoader()

# Load built-in dataset
corpus = loader.load_builtin('minimal')

# Load from file
corpus = loader.load_from_file('data/my_corpus.txt')

# Discover datasets in directory
datasets = loader.load_from_directory('data', pattern='*.txt')

# Get dataset list
available = loader.list_datasets()

# Interactive selection
corpus = loader.interactive_select()

# Analyze corpus
analyzer = DatasetAnalyzer(corpus)
analyzer.print_stats()
```

**Dataset Statistics:**

```
NUMBER OF SENTENCES:     3
TOTAL WORDS:             15
TOTAL CHARACTERS:        87
UNIQUE CHARACTERS:       42
AVERAGE SENTENCE LENGTH: 5.00 words
```

### 3. Weight Initialization (`weight_init.py`)

**Layer Types Supported:**

- `embedding` — Token/position embeddings
- `qkv_proj` — Query/Key/Value projections
- `output_proj` — Attention output projection
- `mlp_expand` — MLP expand layer (C → 4C)
- `mlp_contract` — MLP contract layer (4C → C)
- `lm_head` — Output projection to vocabulary
- `layernorm` — LayerNorm parameters (gamma=1, beta=0)

**Initialization Formula:**

```
scale = 1.0 / sqrt(fan_in)
weight = N(0, scale²)  # Normal distribution
```

**Usage:**

```python
from setup import WeightInitializer, setup_model_initialization

# Get scales for config
scales = setup_model_initialization(model_config)

# Print diagnostic table
print_init_scales_table(model_config)

# Manual initialization
weights = np.random.normal(0, scales['embedding'], size=(vocab, C))
gamma, beta = WeightInitializer.layernorm_init((C,))

# Validate weights
is_valid = WeightInitializer.validate_weight_distribution(
    weights, scales['embedding'], tolerance=0.2
)
```

**Output Example:**

```
INITIALIZATION SCALES TABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer Type                  Scale
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
embedding                   0.10372361
attention_output_proj       0.24999988
lm_head                     0.10372361
mlp_contract                0.07905694
mlp_expand                  0.24999988
qkv_proj                    0.24999988
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4. Training Setup Orchestrator (`training_setup.py`)

**Hyperparameter Presets:**

| Preset | LR | Batch | Epochs | Warmup | Stability |
|--------|-----|-------|--------|--------|-----------|
| **Conservative** | 0.001 | 2 | 5 | 100 | Very stable |
| **Moderate** | 0.01 | 2 | 10 | 500 | Balanced |
| **Aggressive** | 0.1 | 4 | 20 | 200 | Fast but risky |

**Complete Setup Workflow:**

```python
from setup import TrainingSetup

setup = TrainingSetup()

# Run complete interactive wizard
config = setup.run_interactive_setup(use_presets=True)

# Access individual components
print(f"Model: {setup.model_config['name']}")
print(f"Dataset: {setup.dataset_name} ({len(setup.dataset)} sentences)")
print(f"Learning rate: {setup.hyperparams['learning_rate']}")

# Save configuration
setup._save_configuration()

# Later: Load saved configuration
setup.load_configuration('setup/training_config.json')
config = setup.get_complete_config()
```

---

## 🔧 Common Workflows

### Workflow 1: Quick Testing with Defaults

```python
from setup import quickstart_training_setup

config = quickstart_training_setup(interactive=False)

# Use config
model_config = config['model']
corpus = config['dataset']['corpus']
hyperparams = config['hyperparameters']
```

### Workflow 2: Custom Configuration

```python
from setup import ModelConfigBuilder, DatasetLoader

# Build model
builder = ModelConfigBuilder(vocab_size=100)
model_config = builder.interactive_config()
builder.print_summary()

# Select dataset
loader = DatasetLoader()
corpus = loader.interactive_select()
```

### Workflow 3: Load Saved Configuration

```python
from setup import TrainingSetup

setup = TrainingSetup()
config = setup.load_configuration('setup/training_config.json')

# Use for training
model_config = config['model']
corpus = config['dataset']['corpus']
```

### Workflow 4: Estimate VRAM Before Training

```python
from setup import estimate_vram_footprint

estimate = estimate_vram_footprint(model_config)

print(f"Total params: {estimate['total_params']:,}")
print(f"VRAM needed: {estimate['total_mb']:.2f} MB")
print("\nBreakdown:")
for component, count in estimate['breakdown'].items():
    pct = 100 * count / estimate['total_params']
    print(f"  {component}: {count:,} ({pct:.1f}%)")
```

---

## 📊 Configuration File Format

**Example `training_config.json`:**

```json
{
  "model": {
    "name": "Tiny (Testing)",
    "vocab_size": 92,
    "max_len": 8,
    "embedding_dim": 16,
    "num_heads": 2,
    "num_layers": 1,
    "dropout_prob": 0.0,
    "init_scale": 0.02
  },
  "dataset": {
    "name": "minimal",
    "vocab_size": 92,
    "corpus": [
      "cuda training operational step sequence setup complete.",
      "gpu acceleration optimizes matrix transformations.",
      "manual backpropagation gradient loops updating parameter histories."
    ]
  },
  "weight_initialization": {
    "token_embedding": 0.10372361,
    "position_embedding": 0.10372361,
    "qkv_proj": 0.24999988,
    "attention_output_proj": 0.24999988,
    "mlp_expand": 0.24999988,
    "mlp_contract": 0.07905694,
    "lm_head": 0.10372361
  },
  "hyperparameters": {
    "name": "Moderate (Balanced)",
    "learning_rate": 0.01,
    "weight_decay": 0.01,
    "batch_size": 2,
    "num_epochs": 10,
    "warmup_steps": 500,
    "gradient_clip": 1.0,
    "optimizer": "adamw",
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8
  },
  "metadata": {
    "created": "2026-05-23"
  }
}
```

---

## 🔍 Debugging & Troubleshooting

### Check VRAM Requirements

```python
from setup import estimate_vram_footprint

# Tiny config uses ~80KB
# Small config uses ~800KB  
# Medium config uses ~4MB
# All well within GT 730's 1-2GB VRAM

estimate = estimate_vram_footprint(model_config)
if estimate['total_mb'] > 100:
    print("⚠ Warning: Large model, check VRAM!")
```

### Validate Weight Initialization

```python
from setup import WeightInitializer

# Check weights match expected scale
is_valid = WeightInitializer.validate_weight_distribution(
    weights, expected_scale, tolerance=0.2
)

# Print diagnostics
WeightInitializer.print_weight_stats(weights, "embedding_layer")
```

### Verify Dataset Loading

```python
from setup import DatasetAnalyzer

analyzer = DatasetAnalyzer(corpus)
analyzer.print_stats()

# Check uniqueness
vocab_size = len(set(' '.join(corpus)))
print(f"Vocabulary size: {vocab_size}")
```

---

## ✅ Best Practices

1. **Always validate after configuration**
   ```python
   builder.print_summary()
   analyzer.print_stats()
   ```

2. **Save configurations for reproducibility**
   ```python
   setup._save_configuration()
   ```

3. **Use presets for stable training**
   ```python
   config = builder.preset_config('conservative')
   ```

4. **Estimate VRAM before training**
   ```python
   estimate = builder.estimate_memory()
   ```

5. **Check dataset statistics**
   ```python
   analyzer.print_stats()
   ```

---

## 🔗 Integration with Training

Use configured settings in `train.py`:

```python
from setup import quickstart_training_setup
from tokenizer.tokenizer import CharacterGPTTokenizer
from model.gpt import GPTModel

# Get complete configuration
config = quickstart_training_setup()

# Build tokenizer and model
tokenizer = CharacterGPTTokenizer(config['dataset']['corpus'])
model = GPTModel(config['model'])  # Pass as GPTConfig-compatible dict

# Use hyperparameters for training loop
learning_rate = config['hyperparameters']['learning_rate']
num_epochs = config['hyperparameters']['num_epochs']
batch_size = config['hyperparameters']['batch_size']

# Train...
```

---

## 📚 References

- **He Initialization**: He et al., "Delving Deep into Rectifiers" (ICCV 2015)
- **Xavier Initialization**: Glorot & Bengio, "Understanding the difficulty of training deep feedforward networks" (AISTATS 2010)
- **Transformer Architecture**: Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)

---

**Status**: ✅ Complete setup system ready for production use

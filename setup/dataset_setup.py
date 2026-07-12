"""
Dataset Selection and Loading for Kepler GT 730 GPU Training.

Provides:
1. Predefined tiny datasets for quick testing
2. Dataset discovery and loading from data/ directory
3. Vocabulary extraction and statistics
4. Corpus validation and splitting
5. Dataset recommendations based on model config

Target: NVIDIA GeForce GT 730 (1-2GB VRAM constraint)
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from logging_config import logger


# ============================================================================
# PREDEFINED TINY DATASETS FOR TESTING
# ============================================================================

BUILTIN_DATASETS = {
    'minimal': {
        'name': 'Minimal Test Corpus',
        'description': 'Very small 3-sentence corpus for immediate testing',
        'data': [
            "cuda training operational step sequence setup complete.",
            "gpu acceleration optimizes matrix transformations.",
            "manual backpropagation gradient loops updating parameter histories."
        ],
        'vocab_size_estimate': 92,
        'recommended_for': 'unit testing, quick validation'
    },
    'tiny_code': {
        'name': 'Tiny Code Corpus',
        'description': 'Small corpus of Python code snippets',
        'data': [
            "def forward pass through neural network layers",
            "cuda kernel launches gpu accelerated computation",
            "python manages memory allocation on device",
            "tensor operations execute in parallel threads",
            "gpu memory bandwidth limits data transfers",
        ],
        'vocab_size_estimate': 78,
        'recommended_for': 'code generation experiments'
    },
    'tiny_english': {
        'name': 'Tiny English Text',
        'description': 'Small corpus of English sentences',
        'data': [
            "the quick brown fox jumps over the lazy dog",
            "machine learning trains models on large datasets",
            "artificial intelligence mimics human intelligence",
            "neural networks learn patterns from data",
            "deep learning uses multiple layers for feature extraction",
        ],
        'vocab_size_estimate': 85,
        'recommended_for': 'language modeling, NLP experiments'
    },
}


# ============================================================================
# DATASET ANALYZER
# ============================================================================

class DatasetAnalyzer:
    """Analyzes dataset properties and provides statistics."""
    
    def __init__(self, corpus: List[str]):
        """Initialize analyzer with corpus.
        
        Args:
            corpus: List of text strings
        """
        self.corpus = corpus
        self.stats = None
        self._analyze()
    
    def _analyze(self):
        """Compute corpus statistics."""
        text = ' '.join(self.corpus)
        words = text.split()
        chars = list(text)
        
        self.stats = {
            'num_sentences': len(self.corpus),
            'num_words': len(words),
            'num_characters': len(chars),
            'unique_characters': len(set(chars)),
            'avg_sentence_length': len(words) / len(self.corpus) if self.corpus else 0,
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'min_sentence_length': min(len(s.split()) for s in self.corpus) if self.corpus else 0,
            'max_sentence_length': max(len(s.split()) for s in self.corpus) if self.corpus else 0,
        }
    
    def print_stats(self):
        """Print corpus statistics."""
        if self.stats is None:
            return
        
        print("\nDATASET STATISTICS")
        print("-" * 70)
        print(f"Number of sentences:     {self.stats['num_sentences']}")
        print(f"Total words:             {self.stats['num_words']:,}")
        print(f"Total characters:        {self.stats['num_characters']:,}")
        print(f"Unique characters:       {self.stats['unique_characters']}")
        print(f"Average sentence length: {self.stats['avg_sentence_length']:.2f} words")
        print(f"Average word length:     {self.stats['avg_word_length']:.2f} characters")
        print(f"Sentence length range:   {self.stats['min_sentence_length']}-{self.stats['max_sentence_length']} words")
        print("-" * 70 + "\n")


# ============================================================================
# DATASET LOADER
# ============================================================================

class DatasetLoader:
    """Loads and manages datasets from various sources."""
    
    def __init__(self):
        """Initialize loader."""
        self.datasets = {}
        self.current_dataset = None
        logger.info("DatasetLoader initialized")
    
    def load_builtin(self, dataset_name: str) -> List[str]:
        """Load a built-in dataset.
        
        Args:
            dataset_name: Name of built-in dataset ('minimal', 'tiny_code', 'tiny_english')
            
        Returns:
            List of text strings (corpus)
        """
        if dataset_name not in BUILTIN_DATASETS:
            available = list(BUILTIN_DATASETS.keys())
            logger.error(f"Unknown dataset: {dataset_name}. Available: {available}")
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
        
        dataset_info = BUILTIN_DATASETS[dataset_name]
        corpus = dataset_info['data']
        
        self.datasets[dataset_name] = {
            'corpus': corpus,
            'source': 'builtin',
            'info': dataset_info
        }
        self.current_dataset = dataset_name
        
        logger.info(f"Loaded built-in dataset: {dataset_name} ({len(corpus)} sentences)")
        return corpus
    
    def load_from_file(self, filepath: str, dataset_name: Optional[str] = None) -> List[str]:
        """Load dataset from text file.
        
        Args:
            filepath: Path to text file (one sentence per line or one file = one document)
            dataset_name: Optional name for dataset
            
        Returns:
            List of text strings (corpus)
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to split by sentences (periods) or by lines
        if '\n' in content:
            corpus = [line.strip() for line in content.split('\n') if line.strip()]
        else:
            # Split by periods
            corpus = [s.strip() + '.' for s in content.split('.') if s.strip()]
        
        if not corpus:
            logger.error(f"No text content found in {filepath}")
            raise ValueError(f"No text content found in {filepath}")
        
        name = dataset_name or Path(filepath).stem
        self.datasets[name] = {
            'corpus': corpus,
            'source': filepath,
            'info': {
                'name': name,
                'description': f'Loaded from {filepath}',
            }
        }
        self.current_dataset = name
        
        logger.info(f"Loaded dataset from file: {filepath} ({len(corpus)} sentences)")
        return corpus
    
    def load_from_directory(self, directory: str = 'data', 
                           pattern: str = '*.txt') -> Dict[str, List[str]]:
        """Discover and load all datasets from directory.
        
        Args:
            directory: Directory containing text files
            pattern: File pattern to match (default: '*.txt')
            
        Returns:
            Dict mapping dataset names to corpus lists
        """
        data_dir = Path(directory)
        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return {}
        
        datasets = {}
        for filepath in data_dir.glob(pattern):
            try:
                corpus = self.load_from_file(str(filepath), filepath.stem)
                datasets[filepath.stem] = corpus
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        logger.info(f"Discovered {len(datasets)} datasets in {directory}")
        return datasets
    
    def get_corpus(self, dataset_name: Optional[str] = None) -> List[str]:
        """Get corpus for a dataset.
        
        Args:
            dataset_name: Name of dataset (uses current if None)
            
        Returns:
            List of text strings
        """
        name = dataset_name or self.current_dataset
        
        if name is None or name not in self.datasets:
            logger.error(f"Dataset not loaded: {name}")
            raise ValueError(f"Dataset not loaded: {name}")
        
        return self.datasets[name]['corpus']
    
    def list_datasets(self) -> Dict[str, Dict]:
        """List all available datasets (builtin + loaded).
        
        Returns:
            Dict with dataset information
        """
        all_datasets = {}
        
        # Add builtin datasets
        for name, info in BUILTIN_DATASETS.items():
            all_datasets[name] = {
                'type': 'builtin',
                'sentences': len(info['data']),
                'description': info['description'],
            }
        
        # Add loaded datasets
        for name, data in self.datasets.items():
            all_datasets[name] = {
                'type': 'loaded',
                'sentences': len(data['corpus']),
                'source': data['source'],
            }
        
        return all_datasets
    
    def interactive_select(self) -> List[str]:
        """Let user choose dataset interactively.
        
        Returns:
            Selected corpus (list of strings)
        """
        print("\n" + "="*70)
        print("DATASET SELECTION")
        print("="*70)
        
        # List available datasets
        datasets = self.list_datasets()
        dataset_list = list(datasets.keys())
        
        if not dataset_list:
            logger.warning("No datasets available")
            print("No datasets available. Using minimal test corpus...")
            return self.load_builtin('minimal')
        
        print("\nAvailable Datasets:")
        for i, name in enumerate(dataset_list, 1):
            info = datasets[name]
            print(f"  {i}. {name:20s} ({info['sentences']:3d} sentences, {info['type']})")
        
        while True:
            try:
                choice = int(input(f"\nSelect dataset (1-{len(dataset_list)}) [default=1]: ") or "1")
                if 1 <= choice <= len(dataset_list):
                    selected_name = dataset_list[choice - 1]
                    
                    # Load if needed
                    if selected_name not in self.datasets:
                        return self.load_builtin(selected_name)
                    
                    return self.get_corpus(selected_name)
                else:
                    print(f"  ⚠ Please enter a number between 1 and {len(dataset_list)}")
            except ValueError:
                print("  ⚠ Invalid input")


# ============================================================================
# DATASET RECOMMENDATION
# ============================================================================

def recommend_dataset_for_config(model_config: Dict) -> str:
    """Recommend a dataset based on model configuration.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Recommended dataset name
    """
    embedding_dim = model_config.get('embedding_dim', 16)
    num_layers = model_config.get('num_layers', 1)
    max_len = model_config.get('max_len', 8)
    
    # Simple heuristics
    total_params_approx = embedding_dim * num_layers * 1000  # Rough estimate
    
    if total_params_approx < 100000 and max_len <= 16:
        # Tiny model → use minimal dataset
        return 'minimal'
    elif embedding_dim <= 64:
        # Small model → use tiny datasets
        return 'tiny_code'
    else:
        # Larger model → could use larger dataset
        return 'tiny_english'


# ============================================================================
# QUICK START FUNCTION
# ============================================================================

def load_dataset_interactive(model_config: Optional[Dict] = None) -> Tuple[List[str], str]:
    """Interactive dataset loading with recommendations.
    
    Args:
        model_config: Optional model configuration for recommendations
        
    Returns:
        Tuple of (corpus, dataset_name)
    """
    loader = DatasetLoader()
    
    # Show recommendation if config provided
    if model_config:
        recommended = recommend_dataset_for_config(model_config)
        print(f"\n✓ Recommended dataset for your config: {recommended}")
        use_recommended = input("Use recommended dataset? (y/n) [default=y]: ").strip().lower()
        
        if use_recommended != 'n':
            corpus = loader.load_builtin(recommended)
            analyzer = DatasetAnalyzer(corpus)
            analyzer.print_stats()
            return corpus, recommended
    
    # Let user choose
    corpus = loader.interactive_select()
    analyzer = DatasetAnalyzer(corpus)
    analyzer.print_stats()
    
    return corpus, loader.current_dataset

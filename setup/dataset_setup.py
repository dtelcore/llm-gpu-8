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
from paths import DATA_DIR

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


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
    
    def __init__(self, data_dir: str = None, auto_discover: bool = True):
        """Initialize loader.
        
        Args:
            data_dir: Directory to auto-discover .txt datasets from (e.g. 'data')
            auto_discover: If True, immediately scan data_dir so its files show
                up as selectable options alongside the built-in datasets
        """
        self.datasets = {}
        self.current_dataset = None
        self.data_dir = str(data_dir or DATA_DIR)
        # Files found under data_dir but not yet read into memory. Kept separate
        # from self.datasets so large corpora (hundreds of MB) aren't loaded just
        # to populate the selection menu -- only the file actually chosen gets read.
        self._discovered: Dict[str, Path] = {}
        logger.info("DatasetLoader initialized")
        
        if auto_discover:
            self.discover_directory(self.data_dir)
    
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

    def load_by_name(self, dataset_name: str) -> List[str]:
        """Load a dataset by built-in name or discovered file stem."""
        if dataset_name in BUILTIN_DATASETS:
            return self.load_builtin(dataset_name)
        if dataset_name in self._discovered:
            return self.load_from_file(str(self._discovered[dataset_name]), dataset_name)
        if dataset_name in self.datasets:
            return self.get_corpus(dataset_name)
        available = list(BUILTIN_DATASETS.keys()) + list(self._discovered.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    def load_from_file(self, filepath: str, dataset_name: Optional[str] = None) -> List[str]:
        """Load dataset from text file (one document per line for large corpora).
        
        Args:
            filepath: Path to text file (one sentence per line or one file = one document)
            dataset_name: Optional name for dataset
            
        Returns:
            List of text strings (corpus)
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")

        corpus = []
        basename = os.path.basename(filepath)
        logger.info(f"Loading corpus from file: {filepath}")
        with open(filepath, 'r', encoding='utf-8', errors='replace') as handle:
            for raw_line in tqdm(handle, desc=f"Loading corpus: {basename}", unit="line"):
                line = raw_line.strip()
                if line:
                    corpus.append(line)

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
        
        logger.info(f"[OK] Loaded custom dataset: {len(corpus):,} documents from {filepath}")
        return corpus
    
    def discover_directory(self, directory: str = 'data',
                            pattern: str = '*.txt') -> Dict[str, Path]:
        """Discover (but do not read) text files under a directory.

        Only records file paths + cheap metadata (size, first-line preview) so
        that even multi-hundred-MB corpora are safe to "discover" -- the full
        file is only read once a dataset is actually selected via load_from_file.

        Args:
            directory: Directory containing text files
            pattern: File pattern to match (default: '*.txt')

        Returns:
            Dict mapping dataset names (file stem) to their Path
        """
        data_dir = Path(directory)
        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return {}

        discovered = {}
        for filepath in sorted(data_dir.glob(pattern)):
            discovered[filepath.stem] = filepath
            self._discovered[filepath.stem] = filepath

        logger.info(f"Discovered {len(discovered)} dataset file(s) in {directory} (not yet loaded)")
        return discovered

    def load_from_directory(self, directory: str = 'data',
                           pattern: str = '*.txt') -> Dict[str, List[str]]:
        """Discover AND fully load every dataset from directory.

        Prefer discover_directory() for interactive menus -- this eagerly reads
        every matching file into memory, which is expensive for large corpora.

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
        for filepath in sorted(data_dir.glob(pattern)):
            try:
                corpus = self.load_from_file(str(filepath), filepath.stem)
                datasets[filepath.stem] = corpus
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        logger.info(f"Discovered {len(datasets)} datasets in {directory}")
        return datasets

    @staticmethod
    def _preview_file(filepath: Path, max_chars: int = 90) -> str:
        """Read just enough of a file to show a short, unambiguous preview."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                snippet = f.read(max_chars * 2)
            snippet = ' '.join(snippet.split())  # collapse newlines/whitespace
            snippet = snippet[:max_chars]
            return snippet + ('...' if len(snippet) == max_chars else '')
        except Exception:
            return '(preview unavailable)'
    
    def get_corpus(self, dataset_name: Optional[str] = None) -> List[str]:
        """Get corpus for a dataset, lazily loading it from disk if it was
        only discovered (not yet read) so far.
        
        Args:
            dataset_name: Name of dataset (uses current if None)
            
        Returns:
            List of text strings
        """
        name = dataset_name or self.current_dataset
        
        if name is not None and name not in self.datasets and name in self._discovered:
            return self.load_from_file(str(self._discovered[name]), name)
        
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
        
        # Add files discovered under data_dir but not read yet (cheap: size + preview only)
        for name, filepath in self._discovered.items():
            if name in self.datasets:
                continue  # already fully loaded below
            try:
                size_mb = filepath.stat().st_size / (1024 * 1024)
            except OSError:
                size_mb = 0.0
            all_datasets[name] = {
                'type': 'discovered',
                'sentences': None,
                'source': str(filepath),
                'size_mb': size_mb,
                'preview': self._preview_file(filepath),
            }
        
        # Add fully loaded datasets
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
            if info['type'] == 'discovered':
                print(f"  {i}. {name:24s} ({info['size_mb']:.1f} MB on disk, {info['source']})")
                print(f"       preview: \"{info['preview']}\"")
            else:
                source_note = f", {info['source']}" if info['type'] != 'builtin' else ""
                print(f"  {i}. {name:24s} ({info['sentences']:3d} sentences, {info['type']}{source_note})")
        print(f"  0. Load a custom file path (e.g. data/my_corpus.txt)")
        
        while True:
            try:
                raw = input(f"\nSelect dataset (0-{len(dataset_list)}) [default=1]: ") or "1"
                choice = int(raw)
                
                if choice == 0:
                    custom_path = input("  File path: ").strip()
                    return self.load_from_file(custom_path)
                
                if 1 <= choice <= len(dataset_list):
                    selected_name = dataset_list[choice - 1]
                    info = datasets[selected_name]
                    
                    # Load if needed (covers both builtins not yet materialized
                    # and files that were only discovered, not yet read)
                    if info['type'] == 'builtin' and selected_name not in self.datasets:
                        corpus = self.load_builtin(selected_name)
                    else:
                        corpus = self.get_corpus(selected_name)
                    
                    confirm = input(f"  Selected '{selected_name}' ({len(corpus)} sentences). Continue? (y/n) [default=y]: ").strip().lower()
                    if confirm in ('', 'y', 'yes'):
                        return corpus
                    print("  Re-select a dataset below.")
                else:
                    print(f"  ⚠ Please enter a number between 0 and {len(dataset_list)}")
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
    
    if embedding_dim >= 128 and num_layers >= 4 and max_len >= 64:
        return 'tiny_stories'
    if embedding_dim <= 16 and num_layers <= 1 and max_len <= 8:
        return 'minimal'
    if embedding_dim <= 64:
        return 'tiny_code'
    return 'tiny_english'


# ============================================================================
# QUICK START FUNCTION
# ============================================================================

def load_dataset_interactive(model_config: Optional[Dict] = None, data_dir: str = 'data') -> Tuple[List[str], str]:
    """Interactive dataset loading with recommendations.
    
    Args:
        model_config: Optional model configuration for recommendations
        data_dir: Directory auto-scanned for .txt datasets (shown as extra
            options alongside the built-in corpora, e.g. 'data/tiny_stories.txt')
        
    Returns:
        Tuple of (corpus, dataset_name)
    """
    loader = DatasetLoader(data_dir=data_dir)
    
    discovered = [name for name, info in loader.datasets.items()]
    if discovered:
        print(f"\n✓ Found {len(discovered)} dataset file(s) in '{data_dir}/': {', '.join(discovered)}")
    
    # Show recommendation if config provided
    if model_config:
        recommended = recommend_dataset_for_config(model_config)
        print(f"\n✓ Recommended dataset for your config: {recommended}")
        use_recommended = input("Use recommended dataset? (y/n, or 'l' to pick from the list) [default=y]: ").strip().lower()
        
        if use_recommended == 'l':
            corpus = loader.interactive_select()
            analyzer = DatasetAnalyzer(corpus)
            analyzer.print_stats()
            return corpus, loader.current_dataset
        
        if use_recommended != 'n':
            try:
                corpus = loader.load_by_name(recommended)
            except ValueError:
                corpus = loader.load_builtin(recommended)
            analyzer = DatasetAnalyzer(corpus)
            analyzer.print_stats()
            return corpus, loader.current_dataset
    
    # Let user choose (builtins + auto-discovered data_dir files + custom path)
    corpus = loader.interactive_select()
    analyzer = DatasetAnalyzer(corpus)
    analyzer.print_stats()
    
    return corpus, loader.current_dataset

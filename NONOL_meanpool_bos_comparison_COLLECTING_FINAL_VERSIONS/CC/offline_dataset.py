##FINAL REPRODUCIBLE
"""
Offline dataset and datamodule for loading pre-extracted activations.

This module provides Dataset and DataModule classes for loading activations
that were pre-extracted using extract_activations_offline.py, enabling
efficient offline crosscoder training without needing ESM at training time.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


class OfflineActivationDataset(Dataset):
    """
    Dataset for loading pre-extracted activations from .pt files.
    
    This dataset loads activations lazily from chunk files to avoid
    loading all data into memory at once.
    """
    
    def __init__(
        self,
        data_dir: str,
        load_sequences: bool = False,
        load_metadata: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing the extracted activation chunks
            load_sequences: Whether to load and return sequences
            load_metadata: Whether to load and return metadata
        """
        self.data_dir = Path(data_dir)
        self.load_sequences = load_sequences
        self.load_metadata = load_metadata
        
        # Set random seeds for reproducibility
        random_seed = 42
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load manifest
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        self.num_chunks = self.manifest['num_chunks']
        self.total_proteins = self.manifest['total_proteins']
        self.config = self.manifest['config']
        
        # Load config.pt for the full config with tensors
        config_pt_path = self.data_dir / "config.pt"
        if config_pt_path.exists():
            self.full_config = torch.load(config_pt_path)
        else:
            self.full_config = self.config
        
        # Get key configuration values
        self.n_sources = self.config['n_sources']
        self.d_model = self.config['d_model']
        self.bos_vs_meanpool = self.config.get('bos_vs_meanpool', False)
        self.mean_pooled = self.config.get('mean_pooled', False)
        self.layers_to_extract = self.config.get('layers_to_extract', [])
        
        print(f"Loading offline dataset from: {self.data_dir}")
        print(f"  Total proteins: {self.total_proteins:,}")
        print(f"  Number of chunks: {self.num_chunks}")
        print(f"  n_sources: {self.n_sources}")
        print(f"  d_model: {self.d_model}")
        print(f"  BOS vs mean-pool mode: {self.bos_vs_meanpool}")
        
        # Build index: map global index -> (chunk_idx, local_idx)
        self._build_index()
        
        # Cache for loaded chunks (optional, can be memory intensive)
        self._chunk_cache = {}
        self._max_cached_chunks = 2  # Only keep last 2 chunks in memory
    
    def _build_index(self):
        """Build index mapping global protein index to (chunk_idx, local_idx)."""
        self.index = []
        self.chunk_sizes = []
        
        for chunk_idx in range(self.num_chunks):
            chunk_path = self.data_dir / f"chunk_{chunk_idx:05d}.pt"
            if not chunk_path.exists():
                raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
            
            # Load just to get the size (we could also store this in manifest)
            chunk_data = torch.load(chunk_path)
            chunk_size = chunk_data['num_proteins']
            self.chunk_sizes.append(chunk_size)
            
            for local_idx in range(chunk_size):
                self.index.append((chunk_idx, local_idx))
        
        print(f"  Built index with {len(self.index)} entries")
    
    def _load_chunk(self, chunk_idx: int) -> Dict:
        """Load a chunk from disk, with caching."""
        if chunk_idx in self._chunk_cache:
            return self._chunk_cache[chunk_idx]
        
        chunk_path = self.data_dir / f"chunk_{chunk_idx:05d}.pt"
        chunk_data = torch.load(chunk_path)
        
        # Manage cache size
        if len(self._chunk_cache) >= self._max_cached_chunks:
            # Remove oldest cached chunk
            oldest_key = next(iter(self._chunk_cache))
            del self._chunk_cache[oldest_key]
        
        self._chunk_cache[chunk_idx] = chunk_data
        return chunk_data
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a single protein's activations.
        
        Returns:
            activations: [n_sources, d_model] tensor
            metadata: Optional metadata string
            sequence: Optional sequence string
        """
        chunk_idx, local_idx = self.index[idx]
        chunk_data = self._load_chunk(chunk_idx)
        
        activations = chunk_data['activations'][local_idx]  # [n_sources, d_model]
        
        metadata = None
        sequence = None
        
        if self.load_metadata:
            metadata = chunk_data['metadata'][local_idx]
        
        if self.load_sequences:
            sequence = chunk_data['sequences'][local_idx]
        
        return activations, metadata, sequence


class OfflineActivationDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for offline activation loading.
    
    This module provides train/val/test dataloaders for pre-extracted activations.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 512,
        num_workers: int = 0,
        train_split: float = 0.8,
        load_sequences: bool = False,
        load_metadata: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Directory containing the extracted activation chunks
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            train_split: Fraction of data to use for training
            load_sequences: Whether to load and return sequences
            load_metadata: Whether to load and return metadata
            seed: Random seed for train/val split
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.load_sequences = load_sequences
        self.load_metadata = load_metadata
        self.seed = seed
        
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Will be set in setup()
        self.train_dataset = None
        self.test_dataset = None
        self.n_sources = None
        self.d_model = None
        self.full_config = None
        
        # Call setup during init for immediate access to properties
        self.setup()
    
    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets."""
        if self.train_dataset is not None:
            return  # Already setup
        
        # Create full dataset
        dataset = OfflineActivationDataset(
            self.data_dir,
            load_sequences=self.load_sequences,
            load_metadata=self.load_metadata,
        )
        
        # Store key properties
        self.n_sources = dataset.n_sources
        self.d_model = dataset.d_model
        self.full_config = dataset.full_config
        
        # Check if dataset is empty
        if len(dataset) == 0:
            raise ValueError(f"Dataset is empty! Check the data directory: {self.data_dir}")
        
        print(f"Total dataset size: {len(dataset)}")
        
        # Split dataset
        train_size = int(self.train_split * len(dataset))
        test_size = len(dataset) - train_size
        
        if train_size == 0 or test_size == 0:
            raise ValueError(f"Invalid split sizes! train_size: {train_size}, test_size: {test_size}")
        
        # Use generator for reproducible splits
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=generator
        )
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")
    
    @staticmethod
    def offline_collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, List, List]:
        """
        Collate function for offline activations.
        
        This mimics the output format of the online collate function
        to maintain compatibility with the training loop.
        
        Args:
            batch: List of (activations, metadata, sequence) tuples
        
        Returns:
            activations: [batch_size, n_sources, d_model] tensor
            metadata_list: List of metadata (may be None)
            sequences_list: List of sequences (may be None)
        """
        activations_list = []
        metadata_list = []
        sequences_list = []
        
        for activations, metadata, sequence in batch:
            activations_list.append(activations)
            metadata_list.append(metadata)
            sequences_list.append(sequence)
        
        # Stack activations
        stacked_activations = torch.stack(activations_list, dim=0)
        
        return stacked_activations, metadata_list, sequences_list
    
    def train_dataloader(self) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.offline_collate_fn,
            generator=generator,
        )
        print(f"\nTrain DataLoader (Offline):")
        print(f"  Dataset size: {len(self.train_dataset)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Using offline pre-extracted activations")
        return loader
    
    def val_dataloader(self) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.offline_collate_fn,
            generator=generator,
        )
        print(f"\nVal DataLoader (Offline):")
        print(f"  Dataset size: {len(self.test_dataset)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Using offline pre-extracted activations")
        return loader
    
    def test_dataloader(self) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.offline_collate_fn,
            generator=generator,
        )


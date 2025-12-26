##FINAL REPRODUCIBLE
"""
Layer-wise offline dataset for training separate crosscoders per layer class.

This module loads pre-extracted activations but selects only ONE layer class
(first, middle, or last) from each model for comparison.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


class OfflineActivationDatasetLayerwise(Dataset):
    """
    Dataset for loading a single layer class from each model.
    
    Instead of concatenating all 3 layers per model, this selects only
    one layer index (0=first, 1=middle, 2=last) from each model.
    """
    
    def __init__(
        self,
        data_dir: str,
        layer_index: int = 0,  # 0=first, 1=middle, 2=last
        load_sequences: bool = False,
        load_metadata: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing the extracted activation chunks
            layer_index: Which layer to select (0=first, 1=middle, 2=last)
            load_sequences: Whether to load and return sequences
            load_metadata: Whether to load and return metadata
        """
        self.data_dir = Path(data_dir)
        self.layer_index = layer_index
        self.load_sequences = load_sequences
        self.load_metadata = load_metadata
        
        # Validate layer_index
        if layer_index not in [0, 1, 2]:
            raise ValueError(f"layer_index must be 0 (first), 1 (middle), or 2 (last), got {layer_index}")
        
        self.layer_name = ["first", "middle", "last"][layer_index]
        
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
            self.full_config = torch.load(config_pt_path, weights_only=False)
        else:
            self.full_config = self.config
        
        # Multi-model configuration
        self.is_multimodel = self.config.get('extraction_mode') == 'multimodel_first_middle_last'
        if not self.is_multimodel:
            raise ValueError("This dataset requires multi-model extraction format")
        
        self.model_names = self.config.get('models', [])
        self.n_sources = len(self.model_names)  # One source per model
        
        # Get individual model dimensions (single layer, not concatenated)
        self.model_dims = {
            model_name: self.config['model_info'][model_name]['embed_dim']
            for model_name in self.model_names
        }
        
        # d_model = max dimension across models for padding
        self.d_model = max(self.model_dims.values())
        
        # model_groups: each source corresponds to one model [0, 1, 2]
        self.model_groups = list(range(self.n_sources))
        
        print(f"Loading layer-wise offline dataset from: {self.data_dir}")
        print(f"  Layer class: {self.layer_name} (index {self.layer_index})")
        print(f"  Total proteins: {self.total_proteins:,}")
        print(f"  Number of chunks: {self.num_chunks}")
        print(f"  Multi-model mode: {len(self.model_names)} models")
        print(f"  Models: {', '.join(self.model_names)}")
        print(f"  n_sources: {self.n_sources}")
        print(f"  d_model (padded to max): {self.d_model}")
        print(f"  Individual model dimensions (single layer):")
        for model_name, dim in self.model_dims.items():
            print(f"    {model_name}: {dim}")
        print(f"  Model groups: {self.model_groups}")
        
        # Build index: map global index -> (chunk_idx, local_idx)
        self._build_index()
        
        # Cache for loaded chunks
        self._chunk_cache = {}
        self._max_cached_chunks = 2
    
    def _build_index(self):
        """Build index mapping global protein index to (chunk_idx, local_idx)."""
        self.index = []
        self.chunk_sizes = []
        
        for chunk_idx in range(self.num_chunks):
            chunk_path = self.data_dir / f"chunk_{chunk_idx:05d}.pt"
            if not chunk_path.exists():
                raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
            
            # Load just to get the size
            chunk_data = torch.load(chunk_path, weights_only=False)
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
        chunk_data = torch.load(chunk_path, weights_only=False)
        
        # Manage cache size
        if len(self._chunk_cache) >= self._max_cached_chunks:
            oldest_key = next(iter(self._chunk_cache))
            del self._chunk_cache[oldest_key]
        
        self._chunk_cache[chunk_idx] = chunk_data
        return chunk_data
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a single protein's activations for the selected layer class.
        
        Returns:
            activations: [n_sources, d_model] tensor (padded)
            metadata: Optional metadata string
            sequence: Optional sequence string
        """
        chunk_idx, local_idx = self.index[idx]
        chunk_data = self._load_chunk(chunk_idx)
        
        # chunk_data['activations'] is {model_name: [num_proteins, 3, d_model]}
        # We select only self.layer_index from each model
        activations_list = []
        max_dim = self.d_model
        
        for model_name in self.model_names:
            # Get this protein's activations: [3, d_model]
            model_acts = chunk_data['activations'][model_name][local_idx]
            
            # Select only the specified layer: [d_model]
            layer_act = model_acts[self.layer_index]
            
            # Pad to max_dim if necessary
            current_dim = layer_act.shape[0]
            if current_dim < max_dim:
                padding = torch.zeros(max_dim - current_dim, dtype=layer_act.dtype, device=layer_act.device)
                layer_act = torch.cat([layer_act, padding], dim=0)
            
            activations_list.append(layer_act)
        
        # Stack models as sources: [n_models, max_dim]
        activations = torch.stack(activations_list, dim=0)
        
        metadata = None
        sequence = None
        
        if self.load_metadata:
            metadata = chunk_data['metadata'][local_idx]
        
        if self.load_sequences:
            sequence = chunk_data['sequences'][local_idx]
        
        return activations, metadata, sequence


class OfflineActivationDataModuleLayerwise(pl.LightningDataModule):
    """
    Lightning DataModule for layer-wise offline activation loading.
    """
    
    def __init__(
        self,
        data_dir: str,
        layer_index: int = 0,  # 0=first, 1=middle, 2=last
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
            layer_index: Which layer to select (0=first, 1=middle, 2=last)
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            train_split: Fraction of data to use for training
            load_sequences: Whether to load and return sequences
            load_metadata: Whether to load and return metadata
            seed: Random seed for train/val split
        """
        super().__init__()
        self.data_dir = data_dir
        self.layer_index = layer_index
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
        self.is_multimodel = True
        self.model_names = None
        self.model_groups = None
        
        # Call setup during init for immediate access to properties
        self.setup()
    
    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets."""
        if self.train_dataset is not None:
            return  # Already setup
        
        # Create full dataset
        dataset = OfflineActivationDatasetLayerwise(
            self.data_dir,
            layer_index=self.layer_index,
            load_sequences=self.load_sequences,
            load_metadata=self.load_metadata,
        )
        
        # Store key properties
        self.n_sources = dataset.n_sources
        self.d_model = dataset.d_model
        self.full_config = dataset.full_config
        self.model_names = dataset.model_names
        self.model_groups = dataset.model_groups
        
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
        """Collate function for offline activations."""
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
        print(f"\nTrain DataLoader (Offline Layerwise):")
        print(f"  Dataset size: {len(self.train_dataset)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Layer: {['first', 'middle', 'last'][self.layer_index]}")
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
        print(f"\nVal DataLoader (Offline Layerwise):")
        print(f"  Dataset size: {len(self.test_dataset)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Layer: {['first', 'middle', 'last'][self.layer_index]}")
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

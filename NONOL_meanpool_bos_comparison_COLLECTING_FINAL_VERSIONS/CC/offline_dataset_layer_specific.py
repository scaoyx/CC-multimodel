##FINAL REPRODUCIBLE
"""
Layer-specific offline dataset for training crosscoders on arbitrary layer indices.

This module loads pre-extracted activations and selects a SPECIFIC layer index
(e.g., 0, 1, 2, ...) from each model for comparison, not just first/middle/last.
"""
import torch
import pytorch_lightning as pl
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


class OfflineActivationDatasetLayerSpecific(Dataset):
    """
    Dataset for loading a specific layer index from each model.
    
    Unlike the layerwise dataset which uses pre-defined first/middle/last,
    this allows selecting ANY layer index (0, 1, 2, ...) from each model.
    """
    
    def __init__(
        self,
        data_dir: str,
        layer_index: int = 1,  # Specific layer index to extract
        load_sequences: bool = False,
        load_metadata: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing the extracted activation chunks
            layer_index: Specific layer index to select from the stored activations
            load_sequences: Whether to load and return sequences
            load_metadata: Whether to load and return metadata
        """
        self.data_dir = Path(data_dir)
        self.layer_index = layer_index
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
            self.full_config = torch.load(config_pt_path, weights_only=False)
        else:
            self.full_config = self.config
        
        # Multi-model configuration
        extraction_mode = self.config.get('extraction_mode', '')
        self.is_multimodel = extraction_mode in ['multimodel_first_middle_last', 'multimodel_layers_0_1_2', 'multimodel_single_layer_1']
        if not self.is_multimodel:
            raise ValueError(f"This dataset requires multi-model extraction format, got: {extraction_mode}")
        
        # Check if this is single-layer extraction
        self.is_single_layer = extraction_mode == 'multimodel_single_layer_1'
        
        self.model_names = self.config.get('models', [])
        self.n_sources = len(self.model_names)  # One source per model
        
        # Get individual model dimensions (single layer, not concatenated)
        self.model_dims = {
            model_name: self.config['model_info'][model_name]['embed_dim']
            for model_name in self.model_names
        }
        
        # Validate layer_index is within range for all models
        for model_name in self.model_names:
            num_layers = self.config['model_info'][model_name]['num_layers']
            if layer_index > num_layers:
                raise ValueError(
                    f"layer_index {layer_index} exceeds available layers for {model_name} "
                    f"(has {num_layers} layers, indices 0-{num_layers})"
                )
        
        # input_dims is a list of actual dimensions (no padding)
        self.input_dims = [self.model_dims[model_name] for model_name in self.model_names]
        
        # model_groups: each source corresponds to one model [0, 1, 2]
        self.model_groups = list(range(self.n_sources))
        
        print(f"Loading layer-specific offline dataset from: {self.data_dir}")
        print(f"  Specific layer index: {self.layer_index}")
        if self.layer_index == 1:
            print(f"  >>> Comparing SECOND LAYER (index 1) across all models <<<")
        print(f"  Total proteins: {self.total_proteins:,}")
        print(f"  Number of chunks: {self.num_chunks}")
        print(f"  Multi-model mode: {len(self.model_names)} models")
        print(f"  Models: {', '.join(self.model_names)}")
        print(f"  n_sources: {self.n_sources}")
        print(f"  Input dimensions (no padding): {self.input_dims}")
        print(f"  Layer info per model:")
        for model_name in self.model_names:
            num_layers = self.config['model_info'][model_name]['num_layers']
            embed_dim = self.config['model_info'][model_name]['embed_dim']
            print(f"    {model_name}: layer {layer_index}/{num_layers}, dim={embed_dim}")
        print(f"  Model groups: {self.model_groups}")
        
        # Verify the layer data exists in chunks
        # The extraction script saves 3 layers: first (0), middle, last
        # We need to map our desired layer_index to one of these stored positions
        self._determine_stored_layer_mapping()
        
        # Build index: map global index -> (chunk_idx, local_idx)
        self._build_index()
        
        # Cache for loaded chunks
        self._chunk_cache = {}
        self._max_cached_chunks = 2
    
    def _determine_stored_layer_mapping(self):
        """
        Determine which stored layer position contains our desired layer_index.
        
        For single-layer extraction, there's only one layer stored at position 0.
        For multi-layer extraction, we need to find which position contains our desired layer.
        """
        self.layer_position_map = {}
        
        for model_name in self.model_names:
            model_info = self.config['model_info'][model_name]
            
            # Check if this is single-layer extraction
            if model_info.get('single_layer_extraction', False):
                extracted_layer = model_info['extracted_layer_index']
                if self.layer_index == extracted_layer:
                    # Single layer stored at position 0
                    self.layer_position_map[model_name] = 0
                else:
                    raise ValueError(
                        f"Layer index {self.layer_index} not available in extracted data for {model_name}. "
                        f"Only layer {extracted_layer} was extracted (single-layer mode). "
                        f"You need to re-extract activations with the desired layer."
                    )
            else:
                # Multi-layer extraction (old format)
                first_layer = model_info['first_layer']
                middle_layer = model_info['middle_layer']
                last_layer = model_info['last_layer']
                
                # Map desired layer_index to stored position [0, 1, 2]
                if self.layer_index == first_layer:
                    self.layer_position_map[model_name] = 0
                elif self.layer_index == middle_layer:
                    self.layer_position_map[model_name] = 1
                elif self.layer_index == last_layer:
                    self.layer_position_map[model_name] = 2
                else:
                    raise ValueError(
                        f"Layer index {self.layer_index} not available in extracted data for {model_name}. "
                        f"Extracted layers are: {first_layer} (first), {middle_layer} (middle), {last_layer} (last). "
                        f"You need to re-extract activations with the desired layer."
                    )
        
        print(f"  Layer position mapping: {self.layer_position_map}")
    
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
            oldest_chunk = next(iter(self._chunk_cache))
            del self._chunk_cache[oldest_chunk]
        
        self._chunk_cache[chunk_idx] = chunk_data
        return chunk_data
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a single protein's activations for the specified layer index.
        
        Returns:
            activations: Tuple of tensors [d_model_i] with natural dimensions (no padding)
            metadata: Optional metadata string
            sequence: Optional sequence string
        """
        chunk_idx, local_idx = self.index[idx]
        chunk_data = self._load_chunk(chunk_idx)
        
        # chunk_data['activations'] is {model_name: [num_proteins, N, d_model]}
        # where N=1 for single-layer extraction or N=3 for multi-layer extraction
        activations_list = []
        
        for model_name in self.model_names:
            # Get this protein's activations: [N, d_model] where N is 1 or 3
            model_acts = chunk_data['activations'][model_name][local_idx]
            
            # Select the stored position that contains our desired layer
            stored_position = self.layer_position_map[model_name]
            layer_act = model_acts[stored_position]  # [d_model]
            
            # No padding - keep natural dimensions
            activations_list.append(layer_act)
        
        # Return as tuple of tensors with different dimensions
        activations = tuple(activations_list)
        
        metadata = None
        sequence = None
        
        if self.load_metadata:
            metadata = chunk_data['metadata'][local_idx]
        
        if self.load_sequences:
            sequence = chunk_data['sequences'][local_idx]
        
        return activations, metadata, sequence


class OfflineActivationDataModuleLayerSpecific(pl.LightningDataModule):
    """
    Lightning DataModule for layer-specific offline activation loading.
    """
    
    def __init__(
        self,
        data_dir: str,
        layer_index: int = 1,  # Default: second layer (index 1)
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
            layer_index: Specific layer index to select (default: 1 = second layer)
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
        self.input_dims = None
        self.full_config = None
        self.is_multimodel = True
        self.model_names = None
        self.model_groups = None
        
        # Call setup during init for immediate access to properties
        self.setup()
    
    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets."""
        if self.train_dataset is not None:
            return  # Already set up
        
        # Create full dataset
        dataset = OfflineActivationDatasetLayerSpecific(
            self.data_dir,
            layer_index=self.layer_index,
            load_sequences=self.load_sequences,
            load_metadata=self.load_metadata,
        )
        
        # Store key properties
        self.n_sources = dataset.n_sources
        self.input_dims = dataset.input_dims
        self.full_config = dataset.full_config
        self.model_names = dataset.model_names
        self.model_groups = dataset.model_groups
        
        # Check if dataset is empty
        if len(dataset) == 0:
            raise ValueError("Dataset is empty!")
        
        print(f"Total dataset size: {len(dataset)}")
        
        # Split dataset
        train_size = int(self.train_split * len(dataset))
        test_size = len(dataset) - train_size
        
        if train_size == 0 or test_size == 0:
            raise ValueError(f"Train/test split resulted in empty dataset: train={train_size}, test={test_size}")
        
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
    def offline_collate_fn(batch: List[Tuple]) -> Tuple[List[torch.Tensor], List, List]:
        """
        Collate function for offline activations with variable dimensions.
        
        Returns:
            activations: List of tensors [batch, d_i] with different dimensions per source
            metadata_list: List of metadata
            sequences_list: List of sequences
        """
        metadata_list = []
        sequences_list = []
        
        # Collect activations by source
        n_sources = len(batch[0][0])  # Number of sources from first item
        source_batches = [[] for _ in range(n_sources)]
        
        for activations_tuple, metadata, sequence in batch:
            for i, act in enumerate(activations_tuple):
                source_batches[i].append(act)
            metadata_list.append(metadata)
            sequences_list.append(sequence)
        
        # Stack each source independently: [batch, d_i]
        batched_activations = [torch.stack(source_batch, dim=0) for source_batch in source_batches]
        
        return batched_activations, metadata_list, sequences_list
    
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
        print(f"\nTrain DataLoader (Offline Layer-Specific):")
        print(f"  Dataset size: {len(self.train_dataset)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Layer index: {self.layer_index}")
        print(f"  Input dimensions: {self.input_dims}")
        
        # Sample and log batch structure
        sample_batch = next(iter(loader))
        sample_activations = sample_batch[0]
        print(f"  Sample batch structure: List of {len(sample_activations)} tensors")
        for i, act in enumerate(sample_activations):
            print(f"    Source {i}: {tuple(act.shape)}")
        
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
        print(f"\nVal DataLoader (Offline Layer-Specific):")
        print(f"  Dataset size: {len(self.test_dataset)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Layer index: {self.layer_index}")
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

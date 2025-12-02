##FINAL REPRODUCIBLE
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from online_multilayer_dataset import OnlineMultiLayerDataset
from layer_normalization import LayerNormalizationComputer
import torch
import numpy as np
import random

def create_efficient_multilayer_collate_fn(esm_model, alphabet, device, layers_to_extract, normalization_scalars=None, mean_pooled=False):
    """
    Factory function to create an efficient collate function that processes ESM activations at batch level.
    
    Args:
        esm_model: ESM model instance
        alphabet: ESM alphabet
        device: Device to run model on
        layers_to_extract: List of layer indices to extract
        normalization_scalars: Dict mapping layer_idx to normalization scalar (optional)
    
    Returns:
        collate_fn: Function that processes batches efficiently. If mean_pooled=True,
        returns per-protein mean-pooled activations with shape [batch_size, n_sources, d_model].
        Otherwise returns token-level activations concatenated across sequences with shape
        [total_tokens_in_batch, n_sources, d_model].
    """
    batch_converter = alphabet.get_batch_converter()
    
    def efficient_multilayer_collate_fn(batch):
        """
        Efficient collate function for multi-layer online ESM activations.
        Processes entire batch through ESM at once instead of sequence-by-sequence.
        
        Args:
            batch: List of tuples (seq, metadata, idx) - raw sequence data
        
        Returns:
            combined_batch: Tensor of shape [total_tokens, n_sources, d_model] for CrossCoder
            metadata_list: List of metadata for each sequence in batch
            seq_list: List of sequences in batch
        """
        seq_list, metadata_list, idx_list = zip(*batch)
        
        # Prepare batch data for ESM - use metadata as sequence identifiers
        batch_data = [(metadata, seq) for metadata, seq in zip(metadata_list, seq_list)]
        
        # Convert entire batch using batch_converter - this is the key efficiency gain
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        
        # Single ESM forward pass for entire batch - MUCH more efficient!
        with torch.no_grad():
            results = esm_model(
                batch_tokens.to(device), 
                repr_layers=layers_to_extract,
                return_contacts=False
            )
        
        # Process results for each sequence and layer
        n_sources = len(layers_to_extract)
        layer_token_collections = [[] for _ in range(n_sources)]
        
        # Extract embeddings for each sequence in the batch
        for seq_idx in range(len(seq_list)):
            actual_seq_len = len(seq_list[seq_idx])  # Get original sequence length to exclude padding
            for layer_idx_pos, layer_idx in enumerate(layers_to_extract):
                # Extract embeddings for this sequence and layer, removing BOS and EOS tokens
                # Use actual sequence length to exclude padding tokens (consistent with inference)
                embedding = results["representations"][layer_idx][seq_idx, 1:actual_seq_len+1]  # Shape: [actual_len, D]
                
                # If mean_pooled, first pool then (optionally) normalize pooled vector
                if mean_pooled:
                    # Reduce across tokens to a single vector per sequence
                    pooled = embedding.mean(dim=0)  # [D]
                    # Apply layer normalization to pooled vector if scalars are provided
                    if normalization_scalars is not None and layer_idx in normalization_scalars:
                        scalar = normalization_scalars[layer_idx]
                        pooled = pooled / scalar
                    layer_token_collections[layer_idx_pos].append(pooled)
                else:
                    # Token-level path: optionally normalize tokens
                    if normalization_scalars is not None and layer_idx in normalization_scalars:
                        scalar = normalization_scalars[layer_idx]
                        embedding = embedding / scalar  # Normalize to unit average norm
                    layer_token_collections[layer_idx_pos].append(embedding)
        
        # Stack layers to create final batch tensor
        # If mean_pooled, each element in layer_token_collections is a list of [D] vectors per sequence.
        # Otherwise, each element is a list of [T_i, D] tensors to be concatenated across sequences.
        if mean_pooled:
            # For each layer, stack per-sequence vectors -> [batch_size, D]
            stacked_layers = []
            for layer_tokens in layer_token_collections:
                stacked_layer = torch.stack(layer_tokens, dim=0)  # [batch_size, D]
                stacked_layers.append(stacked_layer)
            # Stack across layers -> [batch_size, n_sources, D]
            combined_batch = torch.stack(stacked_layers, dim=1)
        else:
            # Concatenate token embeddings for each layer across all sequences in batch
            # Each element will have shape [total_tokens_in_batch, D]
            concatenated_layers = []
            for layer_tokens in layer_token_collections:
                concatenated_layer = torch.cat(layer_tokens, dim=0)  # [total_tokens, D]
                concatenated_layers.append(concatenated_layer)
            # Stack across layers -> [total_tokens, n_sources, D]
            combined_batch = torch.stack(concatenated_layers, dim=1)
        
        return combined_batch, list(metadata_list), list(seq_list)
    
    return efficient_multilayer_collate_fn


# Keep the old function for backward compatibility (but mark as deprecated)
def multilayer_collate_fn(batch):
    """
    DEPRECATED: Old inefficient collate function.
    This function expects pre-processed embeddings which is inefficient.
    Use create_efficient_multilayer_collate_fn instead.
    """
    raise RuntimeError(
        "This collate function is deprecated and incompatible with the new efficient batch processing. "
        "The dataset now returns raw sequences for batch-level ESM processing."
    )

class OnlineMultiLayerDataModule(pl.LightningDataModule):
    def __init__(self, uniref_file, esm_model, alphabet, device, max_seq_len,
                 batch_size=512, max_samples=50000000, 
                 num_workers=0, layers_to_extract=None, compute_layer_norm=True,
                 norm_samples=500, norm_scalars_path=None, mean_pooled=False):
        super().__init__()
        self.uniref_file = uniref_file
        self.esm_model = esm_model
        self.alphabet = alphabet
        self.device = device
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.num_workers = num_workers
        self.layers_to_extract = layers_to_extract
        self.compute_layer_norm = compute_layer_norm
        self.norm_samples = norm_samples
        self.norm_scalars_path = norm_scalars_path
        self.mean_pooled = mean_pooled

        self.seed = 42  # Hard-coded seed value
        
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Initialize these to None
        self.train_dataset = None
        self.test_dataset = None
        self.n_sources = None
        self.normalization_scalars = None
        
        # Call setup explicitly during initialization
        self.setup()

    def setup(self, stage=None):
        # Add error checking
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            return  # Setup has already been run
        
        # Determine layers to extract
        if self.layers_to_extract is None:
            self.layers_to_extract = list(range(self.esm_model.num_layers + 1))
        
        # Compute or load normalization scalars
        if self.compute_layer_norm:
            if self.norm_scalars_path is not None:
                try:
                    # Try to load existing scalars
                    norm_computer = LayerNormalizationComputer(
                        self.esm_model, self.alphabet, self.device, self.layers_to_extract, self.mean_pooled
                    )
                    self.normalization_scalars = norm_computer.load_scalars(self.norm_scalars_path)
                    print("Loaded existing normalization scalars")
                except:
                    print("Could not load scalars, computing new ones...")
                    self.normalization_scalars = None
            
            if self.normalization_scalars is None:
                # Compute new scalars
                print("Computing layer normalization scalars...")
                norm_computer = LayerNormalizationComputer(
                    self.esm_model, self.alphabet, self.device, self.layers_to_extract, self.mean_pooled
                )
                self.normalization_scalars = norm_computer.compute_normalization_scalars(
                    self.uniref_file, self.max_seq_len, self.norm_samples
                )
                
                # Save scalars if path provided
                if self.norm_scalars_path is not None:
                    norm_computer.save_scalars(self.norm_scalars_path)
        else:
            self.normalization_scalars = None
            print("Layer normalization disabled")
            
        # Create dataset
        dataset = OnlineMultiLayerDataset(
            self.uniref_file, self.esm_model, self.alphabet, self.device,
            self.max_seq_len, self.max_samples, self.layers_to_extract, 
            self.normalization_scalars
        )
        
        # Create efficient collate function for batch-level ESM processing
        self.collate_fn = create_efficient_multilayer_collate_fn(
            self.esm_model, self.alphabet, self.device, 
            self.layers_to_extract, self.normalization_scalars, self.mean_pooled
        )
        
        # Get n_sources from dataset
        self.n_sources = dataset.n_sources
        print(f"Dataset will provide {self.n_sources} sources (layers: {dataset.layers_to_extract})")
        
        # Check if dataset is empty
        if len(dataset) == 0:
            raise ValueError(f"Dataset is empty! Please check the uniref_file path: {self.uniref_file}")
        
        # Log dataset size
        print(f"Total dataset size: {len(dataset)}")
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        # Verify split sizes
        if train_size == 0 or test_size == 0:
            raise ValueError(f"Invalid split sizes! train_size: {train_size}, test_size: {test_size}")
            
        # Use generator for reproducible splits
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.test_dataset = random_split(
            dataset, 
            [train_size, test_size],
            generator=generator
        )
        
        # Log split sizes
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,  # Use efficient batch-level ESM processing
            generator=generator
        )
        print(f"\nTrain DataLoader:")
        print(f"Dataset size: {len(self.train_dataset) if self.train_dataset is not None else 'N/A'}")
        print(f"Batch size: {self.batch_size}")
        print(f"Using efficient batch-level ESM processing")
        return loader

    def val_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,  # Use efficient batch-level ESM processing
            generator=generator 
        )
        print(f"\nVal DataLoader:")
        print(f"Dataset size: {len(self.test_dataset) if self.test_dataset is not None else 'N/A'}")
        print(f"Batch size: {self.batch_size}")
        print(f"Using efficient batch-level ESM processing")
        return loader

    def test_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,  # Use efficient batch-level ESM processing
            generator=generator
        )

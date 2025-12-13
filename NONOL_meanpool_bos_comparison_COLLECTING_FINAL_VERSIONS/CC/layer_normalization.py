##FINAL REPRODUCIBLE
import torch
import numpy as np
from datetime import datetime
import sys
import random
from Bio import SeqIO
import gzip

class LayerNormalizationComputer:
    """
    Computes per-layer normalization scalars to make average token norm = 1 for each layer.
    Also supports BOS vs mean-pool mode where scalars are computed separately for BOS and mean-pooled representations.
    """
    
    def __init__(self, esm_model, alphabet, device, layers_to_extract, mean_pooled=False, 
                 bos_vs_meanpool=False, target_layer=None):
        self.esm_model = esm_model
        self.alphabet = alphabet
        self.device = device
        self.layers_to_extract = layers_to_extract
        self.batch_converter = alphabet.get_batch_converter()
        self.normalization_scalars = None
        self.mean_pooled = mean_pooled
        self.bos_vs_meanpool = bos_vs_meanpool
        self.target_layer = target_layer
        
    def compute_normalization_scalars(self, uniref_file, max_seq_len, n_samples=500, seed=42):
        """
        Compute normalization scalars s_i for each layer i such that 
        dividing activations by s_i results in average norm = 1.
        
        For BOS vs mean-pool mode, computes separate scalars for:
        - 'bos': BOS token representation
        - 'meanpool': Mean-pooled token representations
        
        Args:
            uniref_file: Path to UniRef FASTA file
            max_seq_len: Maximum sequence length to consider
            n_samples: Number of proteins to sample for computing statistics
            seed: Random seed for reproducible sampling
        
        Returns:
            dict: {layer_idx: scalar} or {'bos': scalar, 'meanpool': scalar} mapping
        """
        if self.bos_vs_meanpool:
            print(f"\nComputing BOS vs mean-pool normalization scalars from {n_samples} random proteins...")
            print(f"Target layer: {self.target_layer}")
        else:
            print(f"\nComputing layer normalization scalars from {n_samples} random proteins...")
        
        # Set random seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Collect valid sequences
        valid_sequences = []
        print(f"Loading sequences from {uniref_file}...")
        
        try:
            with gzip.open(uniref_file, "rt") as handle:
                for i, record in enumerate(SeqIO.parse(handle, "fasta")):
                    seq_str = str(record.seq)
                    
                    # Only keep sequences within length limit
                    if len(seq_str) <= max_seq_len:
                        valid_sequences.append(seq_str)
                    
                    # Print progress
                    if (i + 1) % 50000 == 0:
                        print(f"Processed {i+1:,} sequences, found {len(valid_sequences):,} valid sequences")
                    
                    # Stop early if we have enough sequences to sample from
                    if len(valid_sequences) >= min(n_samples * 10, 50000):  # Get 10x more than needed for good sampling
                        print(f"Collected {len(valid_sequences):,} valid sequences for sampling")
                        break
        
        except Exception as e:
            print(f"Error loading sequences: {str(e)}")
            raise
        
        if len(valid_sequences) < n_samples:
            print(f"Warning: Only found {len(valid_sequences)} valid sequences, using all of them")
            sample_sequences = valid_sequences
        else:
            # Randomly sample n_samples sequences
            sample_sequences = random.sample(valid_sequences, n_samples)
        
        print(f"Computing normalization statistics from {len(sample_sequences)} sampled sequences...")
        
        start_time = datetime.now()
        
        # BOS vs mean-pool mode
        if self.bos_vs_meanpool:
            bos_norms = []
            meanpool_norms = []
            
            for i, seq in enumerate(sample_sequences):
                # Prepare data for ESM
                data = [(f"sample_{i}", seq)]
                batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
                
                # Get ESM activations
                with torch.no_grad():
                    results = self.esm_model(
                        batch_tokens.to(self.device),
                        repr_layers=[self.target_layer],
                        return_contacts=False
                    )
                
                actual_seq_len = len(seq)
                full_repr = results["representations"][self.target_layer][0]  # [seq_len_with_special, D]
                
                # BOS token (position 0)
                bos_emb = full_repr[0]  # [D]
                bos_norm = torch.norm(bos_emb).item()
                bos_norms.append(bos_norm)
                
                # Mean-pooled tokens (positions 1 to L, excluding BOS and EOS)
                token_emb = full_repr[1:actual_seq_len+1]  # [L, D]
                meanpool_emb = token_emb.mean(dim=0)  # [D]
                meanpool_norm = torch.norm(meanpool_emb).item()
                meanpool_norms.append(meanpool_norm)
                
                # Print progress
                if (i + 1) % 50 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print(f"Processed {i+1}/{len(sample_sequences)} sequences ({elapsed:.1f}s)")
            
            # Compute normalization scalars
            bos_mean = np.mean(bos_norms)
            bos_std = np.std(bos_norms)
            meanpool_mean = np.mean(meanpool_norms)
            meanpool_std = np.std(meanpool_norms)
            
            normalization_scalars = {
                'bos': bos_mean,
                'meanpool': meanpool_mean
            }
            
            print(f"\nComputed normalization scalars (BOS vs mean-pool mode):")
            print(f"  BOS: mean_norm={bos_mean:.4f}, std_norm={bos_std:.4f}, scalar={bos_mean:.4f}")
            print(f"  Mean-pool: mean_norm={meanpool_mean:.4f}, std_norm={meanpool_std:.4f}, scalar={meanpool_mean:.4f}")
            print(f"  Total proteins: {len(sample_sequences):,}")
            
            self.normalization_scalars = normalization_scalars
            return normalization_scalars
        
        # Original multi-layer mode
        # Collect norms for each layer
        layer_norms = {layer_idx: [] for layer_idx in self.layers_to_extract}
        
        for i, seq in enumerate(sample_sequences):
            # Prepare data for ESM
            data = [(f"sample_{i}", seq)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            
            # Get ESM activations
            with torch.no_grad():
                results = self.esm_model(
                    batch_tokens.to(self.device),
                    repr_layers=self.layers_to_extract,
                    return_contacts=False
                )
            
            # Extract activations for each layer (remove BOS/EOS)
            # Use actual sequence length to exclude padding tokens (consistent with training/inference)
            actual_seq_len = len(seq)
            for layer_idx in self.layers_to_extract:
                activations = results["representations"][layer_idx][0, 1:actual_seq_len+1]  # [actual_len, D]

                if self.mean_pooled:
                    # Compute pooled vector per sequence then its norm
                    pooled = activations.mean(dim=0)  # [D]
                    pooled_norm = torch.norm(pooled).item()
                    layer_norms[layer_idx].append(pooled_norm)
                else:
                    # Compute norm for each token
                    token_norms = torch.norm(activations, dim=1)  # [L]
                    layer_norms[layer_idx].extend(token_norms.cpu().tolist())
            
            # Print progress
            if (i + 1) % 50 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"Processed {i+1}/{len(sample_sequences)} sequences ({elapsed:.1f}s)")
        
        # Compute normalization scalars
        normalization_scalars = {}
        print(f"\nComputed normalization scalars:")
        
        for layer_idx in self.layers_to_extract:
            norms_array = np.array(layer_norms[layer_idx])
            mean_norm = np.mean(norms_array)
            std_norm = np.std(norms_array)
            
            # Scalar to make average norm = 1
            scalar = mean_norm
            normalization_scalars[layer_idx] = scalar
            
            print(f"Layer {layer_idx}: mean_norm={mean_norm:.4f}, std_norm={std_norm:.4f}, scalar={scalar:.4f}")
            print(f"  Total tokens: {len(norms_array):,}")
        
        self.normalization_scalars = normalization_scalars
        return normalization_scalars
    
    def get_normalization_scalars(self):
        """Get the computed normalization scalars."""
        if self.normalization_scalars is None:
            raise ValueError("Normalization scalars not computed yet. Call compute_normalization_scalars() first.")
        return self.normalization_scalars
    
    def save_scalars(self, filepath):
        """Save normalization scalars to file."""
        if self.normalization_scalars is None:
            raise ValueError("No normalization scalars to save.")
        
        torch.save(self.normalization_scalars, filepath)
        print(f"Saved normalization scalars to {filepath}")
    
    def load_scalars(self, filepath):
        """Load normalization scalars from file."""
        self.normalization_scalars = torch.load(filepath)
        print(f"Loaded normalization scalars from {filepath}")
        return self.normalization_scalars

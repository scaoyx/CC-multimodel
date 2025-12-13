##FINAL REPRODUCIBLE
import torch
from torch.utils.data import Dataset
import gzip
from Bio import SeqIO
from datetime import datetime
import sys
import random

class OnlineMultiLayerDataset(Dataset):
    def __init__(self, uniref_file, esm_model, alphabet, device, max_seq_len, 
                 max_samples=50000000, layers_to_extract=None, normalization_scalars=None):
        """
        Dataset that extracts ESM activations online for multiple layers.
        
        Args:
            uniref_file: Path to UniRef FASTA file
            esm_model: ESM model instance
            alphabet: ESM alphabet
            device: Device to run model on
            max_seq_len: Maximum sequence length
            max_samples: Maximum number of samples to load
            layers_to_extract: List of layer indices to extract (if None, extracts all layers)
            normalization_scalars: Dict mapping layer_idx to normalization scalar (optional)
        """
        
        # Set random seeds for reproducibility
        random_seed = 42  # Hardcoded random seed
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.esm_model = esm_model
        self.alphabet = alphabet
        self.device = device
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples
        self.normalization_scalars = normalization_scalars
        self.batch_converter = alphabet.get_batch_converter()
        
        # Determine which layers to extract
        if layers_to_extract is None:
            # Extract all layers (ESM models typically have num_layers + 1 representations)
            self.layers_to_extract = list(range(esm_model.num_layers + 1))
        else:
            self.layers_to_extract = layers_to_extract
            
        self.n_sources = len(self.layers_to_extract)
        print(f"Will extract activations from {self.n_sources} layers: {self.layers_to_extract}")
        
        if self.normalization_scalars is not None:
            print(f"Using layer normalization scalars: {self.normalization_scalars}")
        else:
            print("No layer normalization will be applied")
        
        # Store sequences and metadata (same as original AA_dataset.py)
        self.sequences = []
        self.metadata = []

        print(f"Loading sequences from {uniref_file}...")
        print(f"Max sequence length: {max_seq_len}")
        print(f"Max samples to load: {max_samples:,}")
        sys.stdout.flush()

        total_sequences = 0
        loaded_sequences = 0
        too_long_sequences = 0
        last_print_time = datetime.now()

        try:
            with gzip.open(uniref_file, "rt") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    total_sequences += 1
                    seq_str = str(record.seq)
                    
                    # Only filter based on sequence length
                    if len(seq_str) > max_seq_len:
                        too_long_sequences += 1
                        continue

                    # Load all sequences that pass length filter
                    self.sequences.append(seq_str)
                    self.metadata.append(record.description)
                    loaded_sequences += 1

                    # Updated progress printing
                    current_time = datetime.now()
                    if (current_time - last_print_time).total_seconds() >= 10 or loaded_sequences % 100000 == 0:
                        computed_sum = loaded_sequences + too_long_sequences
                        breakdown = (
                            f"\rProcessed: {total_sequences:,} "
                            f"[Loaded: {loaded_sequences:,} + "
                            f"Too long: {too_long_sequences:,} = {computed_sum:,}]"
                        )
                        
                        # Add verification
                        if computed_sum != total_sequences:
                            breakdown += f" WARNING: Sum mismatch! Diff: {total_sequences - computed_sum:,}"
                        
                        print(breakdown, end="")
                        sys.stdout.flush()
                        last_print_time = current_time
                    
                    if loaded_sequences >= self.max_samples:
                        print(f"\nReached the maximum number of sequences ({loaded_sequences:,}). Stopping sequence loading.")
                        break

        except Exception as e:
            print(f"Error loading UniRef dataset: {str(e)}")
            raise

        print(f"\n\nDataset loading complete.")
        print(f"Total sequences processed: {total_sequences:,}")
        print(f"Sequences too long: {too_long_sequences:,}")
        print(f"Sequences loaded: {loaded_sequences:,}")
        sys.stdout.flush()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Return raw sequence data for batch-level ESM processing.
        
        Returns:
            tuple: (seq, metadata, idx) - raw data for batch processing
        """
        seq = self.sequences[idx]
        metadata = self.metadata[idx]
        
        # Return raw data - ESM processing will happen in collate_fn
        return seq, metadata, idx

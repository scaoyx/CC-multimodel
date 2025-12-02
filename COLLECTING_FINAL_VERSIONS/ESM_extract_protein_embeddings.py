#!/usr/bin/env python3
"""
Extract raw mean-pooled ESM activations per layer.

This script:
1. Loads protein sequences from a TSV file (with 'Sequence' column)
2. Extracts ESM activations for each protein
3. Mean-pools each layer's activations (without normalization)
4. Optionally applies ReLU to mean-pooled activations (--relu 1)
5. Saves one .pt file per layer containing all proteins' activations for that layer

Usage:
    python extract_raw_esm_activations.py \
        --tsv_file path/to/proteins.tsv \
        --esm_model esm2_t12_35M_UR50D \
        --max_seq_len 1024 \
        --output_dir esm_activations/ \
        --device cuda:0 \
        --relu 1
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# Import ESM
try:
    import esm
except ImportError:
    print("ESM library not found. Install with: pip install fair-esm")
    sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract raw mean-pooled ESM activations per layer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input files
    parser.add_argument("--tsv_file", type=str, required=True, 
                        help="Path to TSV file with 'Sequence' column")
    
    # ESM model parameters
    parser.add_argument("--esm_model", type=str, default="esm2_t12_35M_UR50D",
                        choices=["esm2_t33_650M_UR50D", "esm2_t6_8M_UR50D", 
                                "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D"],
                        help="ESM model to use")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="Maximum sequence length to process")
    parser.add_argument("--layers_to_extract", type=int, nargs='+', default=None,
                        help="Specific layers to extract (if None, uses all layers)")
    
    # Processing parameters
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('cuda', 'cuda:0', 'cpu', etc.)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to first N proteins in TSV (for testing)")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory to save per-layer activation files")
    parser.add_argument("--save_metadata", type=int, default=1, choices=[0, 1],
                        help="Save metadata (sequences, labels) along with activations")
    parser.add_argument("--relu", type=int, default=0, choices=[0, 1],
                        help="Apply ReLU to mean-pooled activations (0=no, 1=yes)")

    return parser.parse_args()


def load_esm_model(esm_model_name: str, device: torch.device):
    """Load ESM model"""
    print(f"Loading ESM model: {esm_model_name}")
    
    model_dict = {
        "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
    }
    
    esm_model, alphabet = model_dict[esm_model_name]()
    esm_model = esm_model.to(device)
    esm_model.eval()
    
    print(f"ESM model loaded:")
    print(f"  - Number of layers: {esm_model.num_layers}")
    print(f"  - Embedding dimension: {esm_model.embed_dim}")
    
    return esm_model, alphabet


def load_protein_sequences(tsv_path: str, max_seq_len: int, max_samples: Optional[int] = None) -> Tuple[List[str], List[str], Dict]:
    """Load protein sequences from TSV file"""
    print(f"Loading protein sequences from: {tsv_path}")
    
    if not Path(tsv_path).exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    
    # Read TSV
    df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
    if 'Sequence' not in df.columns:
        raise ValueError("TSV missing required 'Sequence' column")
    
    total_rows = len(df)
    print(f"Total rows in TSV: {total_rows}")
    
    # Limit samples if requested
    if max_samples is not None:
        df = df.iloc[:max_samples].copy()
        print(f"Limited to first {max_samples} rows")
    
    # Filter by sequence length
    seq_series = df['Sequence'].astype(str)
    valid_mask = seq_series.str.len() <= max_seq_len
    skipped_too_long = int((~valid_mask).sum())
    
    if skipped_too_long > 0:
        print(f"Skipping {skipped_too_long} sequences longer than {max_seq_len}")
    
    df_filtered = df[valid_mask].reset_index(drop=True)
    sequences = df_filtered['Sequence'].astype(str).tolist()
    
    # Create metadata labels
    if 'Entry' in df_filtered.columns and 'Entry Name' in df_filtered.columns:
        entry = df_filtered['Entry'].astype(str).tolist()
        entry_name = df_filtered['Entry Name'].astype(str).tolist()
        labels = [f"{e}|{n}" for e, n in zip(entry, entry_name)]
    elif 'Entry' in df_filtered.columns:
        labels = df_filtered['Entry'].astype(str).tolist()
    else:
        labels = [f"protein_{i}" for i in range(len(sequences))]
    
    stats = {
        'total_rows_in_tsv': total_rows,
        'sequences_after_length_filter': len(sequences),
        'sequences_skipped_too_long': skipped_too_long,
        'max_seq_len_used': max_seq_len,
    }
    
    print(f"Loaded {len(sequences)} valid protein sequences")
    return sequences, labels, stats


def extract_mean_pooled_esm_activations(
    sequence: str, 
    label: str,
    esm_model, 
    alphabet, 
    device: torch.device,
    layers_to_extract: List[int],
) -> Dict[int, torch.Tensor]:
    """
    Extract mean-pooled ESM activations for a single protein sequence.
    
    Args:
        sequence: Amino acid sequence
        label: Protein identifier
        esm_model: ESM model
        alphabet: ESM alphabet
        device: Compute device
        layers_to_extract: List of layer indices to extract
    
    Returns:
        Dictionary mapping layer_idx -> mean-pooled activation vector [d_model]
    """
    batch_converter = alphabet.get_batch_converter()
    
    # Prepare data
    data = [(label, sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    # Extract activations
    with torch.no_grad():
        results = esm_model(
            batch_tokens.to(device),
            repr_layers=layers_to_extract,
            return_contacts=False
        )
    
    # Process each layer: extract tokens (excluding BOS/EOS) and mean pool
    layer_activations = {}
    seq_len = len(sequence)
    
    for layer_idx in layers_to_extract:
        # Remove BOS and EOS tokens: [1:seq_len+1]
        embedding = results["representations"][layer_idx][0, 1:seq_len+1]  # [seq_len, d_model]
        
        # Mean pool across sequence (NO NORMALIZATION)
        pooled = embedding.mean(dim=0)  # [d_model]
        
        layer_activations[layer_idx] = pooled.cpu()
    
    return layer_activations


def main():
    args = parse_arguments()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("Raw ESM Activation Extractor (Mean-Pooled Per Layer)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"TSV file: {args.tsv_file}")
    print(f"ESM model: {args.esm_model}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Output directory: {args.output_dir}")
    print(f"ReLU activation: {'Yes' if args.relu == 1 else 'No'}")
    print("=" * 60)
    
    # Load ESM model
    esm_model, alphabet = load_esm_model(args.esm_model, device)
    
    # Determine layers to extract
    if args.layers_to_extract is None:
        layers_to_extract = list(range(esm_model.num_layers + 1))
    else:
        layers_to_extract = args.layers_to_extract
    
    print(f"Will extract from {len(layers_to_extract)} layers: {layers_to_extract}")
    
    # Load protein sequences
    sequences, labels, stats = load_protein_sequences(
        args.tsv_file, args.max_seq_len, args.max_samples
    )
    
    if len(sequences) == 0:
        print("No valid sequences to process!")
        return
    
    # Initialize storage for each layer
    # layer_activations_dict[layer_idx] = list of tensors, one per protein
    layer_activations_dict = {layer_idx: [] for layer_idx in layers_to_extract}
    processed_labels = []
    processed_sequences = []
    
    # Process each protein
    print(f"\nProcessing {len(sequences)} proteins...")
    
    for i, (sequence, label) in enumerate(tqdm(zip(sequences, labels), total=len(sequences), desc="Extracting activations")):
        try:
            # Extract mean-pooled activations for all layers
            protein_layer_acts = extract_mean_pooled_esm_activations(
                sequence, label, esm_model, alphabet, device, layers_to_extract
            )
            
            # Apply ReLU if requested (set negative values to 0)
            if args.relu == 1:
                for layer_idx in layers_to_extract:
                    protein_layer_acts[layer_idx] = torch.relu(protein_layer_acts[layer_idx])
            
            # Store each layer's activation
            for layer_idx in layers_to_extract:
                layer_activations_dict[layer_idx].append(protein_layer_acts[layer_idx])
            
            processed_labels.append(label)
            processed_sequences.append(sequence)
            
        except Exception as e:
            print(f"\nError processing protein {i} ({label}): {e}")
            continue
        
        # Progress update
        if (i + 1) % 100 == 0:
            print(f"\nProcessed {i + 1}/{len(sequences)} proteins")
    
    if len(processed_labels) == 0:
        print("No proteins were successfully processed!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(processed_labels)} proteins")
    print(f"\nSaving per-layer activation files...")
    
    # Save each layer as a separate file
    for layer_idx in layers_to_extract:
        # Stack all proteins' activations for this layer
        layer_tensor = torch.stack(layer_activations_dict[layer_idx], dim=0)  # [n_proteins, d_model]
        
        # Prepare output data for this layer
        output_data = {
            'embeddings': layer_tensor,
            'layer_idx': layer_idx,
            'processing_info': {
                'esm_model': args.esm_model,
                'layer_idx': layer_idx,
                'max_seq_len': args.max_seq_len,
                'embedding_dim': layer_tensor.shape[1],
                'n_proteins_processed': len(processed_labels),
                'timestamp': datetime.now().isoformat(),
                'normalization_applied': False,  # Explicitly note no normalization
                'relu_applied': bool(args.relu),  # Whether ReLU was applied
            }
        }
        
        # Add metadata if requested
        if args.save_metadata:
            output_data['metadata'] = {
                'labels': processed_labels,
                'sequences': processed_sequences,
            }
            output_data.update(stats)
        
        # Save this layer's file
        layer_filename = f"layer_{layer_idx}_activations.pt"
        layer_filepath = output_dir / layer_filename
        torch.save(output_data, layer_filepath)
        
        file_size_mb = layer_filepath.stat().st_size / (1024*1024)
        print(f"  Layer {layer_idx:2d}: {layer_tensor.shape} -> {layer_filename} ({file_size_mb:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("Extraction complete!")
    print(f"Saved {len(layers_to_extract)} layer files to: {args.output_dir}")
    print(f"Each file contains {len(processed_labels)} protein activations")
    print(f"Embedding dimension: {layer_tensor.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()


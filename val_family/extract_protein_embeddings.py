#!/usr/bin/env python3
"""
Extract protein embeddings using trained crosscoder model.

This script:
1. Loads protein sequences from a TSV file (with 'Sequence' column)
2. Extracts ESM activations for each protein using the same method as training
3. Passes activations through a trained crosscoder model
4. Extracts hidden activations (before decoding) and max-pools them per protein
5. Saves one vector per protein that meets length criteria

Usage:
    python extract_protein_embeddings.py \
        --tsv_file path/to/proteins.tsv \
        --checkpoint_path path/to/trained_model.ckpt \
        --esm_model esm2_t6_8M_UR50D \
        --max_seq_len 1024 \
        --output_file protein_embeddings.pt
"""

import os
import sys
import argparse
import pickle
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# Import ESM and other required modules
try:
    import esm
except ImportError:
    print("ESM library not found. Install with: pip install fair-esm")
    sys.exit(1)

# Import the crosscoder classes
from sparse_auto_script import LitLit
from layer_normalization import LayerNormalizationComputer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract protein embeddings using trained crosscoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input files
    parser.add_argument("--tsv_file", type=str, required=True, 
                        help="Path to TSV file with 'Sequence' column")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained crosscoder checkpoint (.ckpt file)")
    
    # ESM model parameters
    parser.add_argument("--esm_model", type=str, default="esm2_t6_8M_UR50D",
                        choices=["esm2_t33_650M_UR50D", "esm2_t6_8M_UR50D", 
                                "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D"],
                        help="ESM model to use (should match training)")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="Maximum sequence length to process")
    parser.add_argument("--layers_to_extract", type=int, nargs='+', default=None,
                        help="Specific layers to extract (if None, uses all layers)")
    
    # Layer normalization (should match training settings)
    parser.add_argument("--use_layer_norm", type=int, default=1, choices=[0, 1],
                        help="Apply layer normalization (1=yes, 0=no, should match training)")
    parser.add_argument("--norm_scalars_path", type=str, default=None,
                        help="Path to load normalization scalars (if used during training)")
    parser.add_argument("--norm_samples", type=int, default=500,
                        help="Number of proteins to sample for computing normalization (if computing new)")
    
    # Processing parameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for ESM processing (1 for per-protein processing)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('cuda', 'cuda:0', 'cpu', etc.)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to first N proteins in TSV (for testing)")
    parser.add_argument("--pooling_method", type=str, default="max", 
                        choices=["max", "mean"],
                        help="Pooling method for sequence aggregation (default: max)")
    parser.add_argument("--mean_pooled", type=int, default=0, choices=[0, 1],
                        help="If 1, mean-pool per-layer activations before passing to crosscoder and use pooled-level normalization")
    
    # Output
    parser.add_argument("--output_file", type=str, default="protein_embeddings.pt",
                        help="Output file to save protein embeddings")
    parser.add_argument("--save_metadata", type=int, default=1, choices=[0, 1],
                        help="Save metadata (sequences, labels) along with embeddings")

    return parser.parse_args()


def load_crosscoder_from_checkpoint(checkpoint_path: str, device: torch.device) -> LitLit:
    """Load trained crosscoder model from checkpoint"""
    print(f"Loading crosscoder from checkpoint: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load using PyTorch Lightning's method
    model = LitLit.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    
    # Get model info
    hparams = model.hparams
    print(f"Loaded crosscoder:")
    print(f"  - Hidden dimension: {hparams.hidden_dim}")
    print(f"  - Input dimensions: {hparams.input_dims}")
    print(f"  - Number of sources: {len(hparams.input_dims)}")
    print(f"  - K (sparsity): {hparams.k}")
    
    return model


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


def setup_layer_normalization(args, esm_model, alphabet, device) -> Optional[Dict[int, float]]:
    """Setup layer normalization if needed"""
    if not args.use_layer_norm:
        print("Layer normalization disabled")
        return None
    
    # Determine layers to extract
    if args.layers_to_extract is None:
        layers_to_extract = list(range(esm_model.num_layers + 1))
    else:
        layers_to_extract = args.layers_to_extract
    
    print(f"Setting up layer normalization for layers: {layers_to_extract}")
    
    # Try to load existing scalars
    if args.norm_scalars_path and Path(args.norm_scalars_path).exists():
        print(f"Loading normalization scalars from: {args.norm_scalars_path}")
        norm_computer = LayerNormalizationComputer(esm_model, alphabet, device, layers_to_extract, bool(args.mean_pooled))
        return norm_computer.load_scalars(args.norm_scalars_path)
    
    # Need to compute new scalars - requires a UniRef file
    print("Warning: No normalization scalars provided and none found at specified path.")
    print("Layer normalization will be disabled for this run.")
    print("To use layer normalization, either:")
    print("1. Provide --norm_scalars_path pointing to existing scalars, or")
    print("2. Use the layer_normalization.py script to compute scalars first")
    
    return None


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


def extract_esm_activations_for_protein(
    sequence: str, 
    label: str,
    esm_model, 
    alphabet, 
    device: torch.device,
    layers_to_extract: List[int],
    normalization_scalars: Optional[Dict[int, float]] = None,
    mean_pooled: bool = False
) -> torch.Tensor:
    """
    Extract ESM activations for a single protein sequence.
    Returns tensor of shape [seq_len, n_sources, d_model] like during training.
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
    
    # Process each layer (same as OnlineMultiLayerDataset)
    layer_embeddings = []
    seq_len = len(sequence)
    
    for layer_idx in layers_to_extract:
        # Remove BOS and EOS tokens: [1:seq_len+1]
        embedding = results["representations"][layer_idx][0, 1:seq_len+1]  # [seq_len, d_model]
        
        if mean_pooled:
            # Mean pool first, then apply pooled-level normalization if provided
            pooled = embedding.mean(dim=0)  # [d_model]
            if normalization_scalars is not None and layer_idx in normalization_scalars:
                scalar = normalization_scalars[layer_idx]
                pooled = pooled / scalar
            layer_embeddings.append(pooled)
        else:
            # Apply token-level normalization if available, then keep token sequence
            if normalization_scalars is not None and layer_idx in normalization_scalars:
                scalar = normalization_scalars[layer_idx]
                embedding = embedding / scalar
            layer_embeddings.append(embedding)
    
    # Stack per-layer outputs
    if mean_pooled:
        # [n_sources, d_model] -> [1, n_sources, d_model] to mimic batch of 1
        multi_layer_activations = torch.stack(layer_embeddings, dim=0).unsqueeze(0)
    else:
        # [seq_len, n_sources, d_model]
        multi_layer_activations = torch.stack(layer_embeddings, dim=1)
    
    return multi_layer_activations


def extract_hidden_activations_from_crosscoder(
    activations: torch.Tensor, 
    crosscoder_model: LitLit,
    mean_pooled: bool = False
) -> torch.Tensor:
    """
    Pass activations through crosscoder and extract hidden activations.
    
    Args:
        activations: [seq_len, n_sources, d_model] - ESM activations for one protein
        crosscoder_model: Trained crosscoder model
    
    Returns:
        hidden_acts: [seq_len, hidden_dim] - Sparse hidden activations after TopK
    """
    # The crosscoder expects [batch, n_sources, d_model]
    # mean_pooled=False: we have [seq_len, n_sources, d_model] where seq_len acts as batch dimension
    # mean_pooled=True: we have [1, n_sources, d_model]
    
    with torch.no_grad():
        # First encode to get combined latents
        combined_latents = crosscoder_model.model.encode(activations)
        # combined_latents shape: [seq_len, hidden_dim]
        
        # Then apply sparse activation (TopK) to get the sparse hidden activations
        sparse_acts, _, _ = crosscoder_model.model.sparse_activation.forward(combined_latents)
        # sparse_acts shape: [seq_len, hidden_dim] or [1, hidden_dim]
    
    return sparse_acts


def pool_hidden_activations(hidden_acts: torch.Tensor, pooling_method: str = "max") -> torch.Tensor:
    """
    Pool hidden activations across sequence length.
    
    Args:
        hidden_acts: [seq_len, hidden_dim]
        pooling_method: "max" for max pooling or "mean" for mean pooling
    
    Returns:
        pooled: [hidden_dim] - Pooled representation
    """
    if pooling_method == "max":
        # Max pool across sequence dimension (dim=0)
        pooled, _ = torch.max(hidden_acts, dim=0)
    elif pooling_method == "mean":
        # Mean pool across sequence dimension (dim=0)
        pooled = torch.mean(hidden_acts, dim=0)
    else:
        raise ValueError(f"Unsupported pooling method: {pooling_method}. Choose 'max' or 'mean'.")
    
    return pooled


def main():
    args = parse_arguments()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("Protein Embedding Extractor using Trained Crosscoder")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"TSV file: {args.tsv_file}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"ESM model: {args.esm_model}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Pooling method: {args.pooling_method}")
    print(f"Output file: {args.output_file}")
    print("=" * 60)
    
    # Load models
    crosscoder_model = load_crosscoder_from_checkpoint(args.checkpoint_path, device)
    esm_model, alphabet = load_esm_model(args.esm_model, device)
    
    # Determine layers to extract (should match what was used during training)
    if args.layers_to_extract is None:
        layers_to_extract = list(range(esm_model.num_layers + 1))
    else:
        layers_to_extract = args.layers_to_extract
    
    print(f"Will extract from layers: {layers_to_extract}")
    
    # Verify layer count matches crosscoder expectation
    expected_n_sources = len(crosscoder_model.hparams.input_dims)
    actual_n_sources = len(layers_to_extract)
    if actual_n_sources != expected_n_sources:
        raise ValueError(
            f"Layer count mismatch! Crosscoder expects {expected_n_sources} sources, "
            f"but {actual_n_sources} layers specified. "
            f"Use --layers_to_extract to specify the exact layers used during training."
        )
    
    # Setup layer normalization
    normalization_scalars = setup_layer_normalization(args, esm_model, alphabet, device)
    
    # Load protein sequences
    sequences, labels, stats = load_protein_sequences(
        args.tsv_file, args.max_seq_len, args.max_samples
    )
    
    if len(sequences) == 0:
        print("No valid sequences to process!")
        return
    
    # Process each protein
    print(f"\nProcessing {len(sequences)} proteins...")
    
    protein_embeddings = []
    processed_labels = []
    processed_sequences = []
    
    for i, (sequence, label) in enumerate(tqdm(zip(sequences, labels), total=len(sequences), desc="Processing proteins")):
        try:
            # Extract ESM activations for this protein
            esm_activations = extract_esm_activations_for_protein(
                sequence, label, esm_model, alphabet, device, 
                layers_to_extract, normalization_scalars, bool(args.mean_pooled)
            )
            # esm_activations shape: [seq_len, n_sources, d_model]
            
            # Pass through crosscoder to get hidden activations
            hidden_acts = extract_hidden_activations_from_crosscoder(
                esm_activations, crosscoder_model, bool(args.mean_pooled)
            )
            # hidden_acts shape: [seq_len, hidden_dim] or [1, hidden_dim]
            
            # Pool to get single vector per protein
            if bool(args.mean_pooled):
                # Already pooled at input; remove batch dim
                protein_embedding = hidden_acts.squeeze(0)
            else:
                protein_embedding = pool_hidden_activations(hidden_acts, args.pooling_method)
            # protein_embedding shape: [hidden_dim]
            
            # Store results
            protein_embeddings.append(protein_embedding.cpu())
            processed_labels.append(label)
            processed_sequences.append(sequence)
            
        except Exception as e:
            print(f"Error processing protein {i} ({label}): {e}")
            continue
        
        # Progress update
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(sequences)} proteins")
    
    if len(protein_embeddings) == 0:
        print("No proteins were successfully processed!")
        return
    
    # Stack all embeddings
    all_embeddings = torch.stack(protein_embeddings, dim=0)  # [n_proteins, hidden_dim]
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(protein_embeddings)} proteins")
    print(f"Final embeddings shape: {all_embeddings.shape}")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")
    
    # Prepare output data
    output_data = {
        'embeddings': all_embeddings,
        'processing_info': {
            'checkpoint_path': args.checkpoint_path,
            'esm_model': args.esm_model,
            'layers_extracted': layers_to_extract,
            'max_seq_len': args.max_seq_len,
            'use_layer_norm': bool(args.use_layer_norm),
            'pooling_method': args.pooling_method,
            'hidden_dim': all_embeddings.shape[1],
            'n_proteins_processed': len(protein_embeddings),
            'timestamp': datetime.now().isoformat(),
        }
    }
    
    # Add metadata if requested
    if args.save_metadata:
        output_data['metadata'] = {
            'labels': processed_labels,
            'sequences': processed_sequences,
        }
        output_data.update(stats)
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving results to: {args.output_file}")
    torch.save(output_data, args.output_file)
    
    print("\nExtraction complete!")
    print(f"Saved {len(protein_embeddings)} protein embeddings")
    print(f"Each embedding is a {all_embeddings.shape[1]}-dimensional vector")
    print(f"Total file size: ~{output_path.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()

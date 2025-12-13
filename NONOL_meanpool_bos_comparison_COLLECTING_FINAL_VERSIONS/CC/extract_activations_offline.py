#!/usr/bin/env python3
"""
Extract and store ESM activations for offline crosscoder training.

This script extracts activations EXACTLY as done during online training,
but stores them as .pt files for efficient offline loading.

The activations are stored in chunks to avoid memory issues and enable
efficient data loading during training.

Usage:
    python extract_activations_offline.py \
        --uniref_file uniref50.fasta.gz \
        --esm_model esm2_t12_35M_UR50D \
        --max_seq_len 1024 \
        --num_proteins 500000 \
        --batch_size 128 \
        --bos_vs_meanpool 1 \
        --target_layer 11 \
        --norm_scalars_path BOS_vs_Meanpool_norm_scalars/bos_vs_meanpool_layer11_norm_scalars.pt \
        --output_dir offline_activations/bos_vs_meanpool_layer11 \
        --proteins_per_file 10000 \
        --cuda_device 0
"""

import os
import sys
import argparse
import gzip
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm

try:
    import esm
except ImportError:
    print("ESM library not found. Install with: pip install fair-esm")
    sys.exit(1)

from layer_normalization import LayerNormalizationComputer


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Extract ESM activations for offline crosscoder training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--uniref_file', type=str, required=True,
                        help='Path to UniRef FASTA file (.gz)')
    parser.add_argument('--esm_model', type=str, default="esm2_t12_35M_UR50D",
                        choices=["esm2_t33_650M_UR50D", "esm2_t6_8M_UR50D",
                                 "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D"],
                        help='ESM model to use for activation extraction')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--num_proteins', type=int, default=500000,
                        help='Number of proteins to extract activations for')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for ESM processing')

    # Mode selection (matching main_script_online.py)
    parser.add_argument('--bos_vs_meanpool', type=int, default=0, choices=[0, 1],
                        help='If 1, extract BOS token vs mean-pooled representations from a single layer')
    parser.add_argument('--target_layer', type=int, default=None,
                        help='Target layer for bos_vs_meanpool mode (default: last layer)')
    parser.add_argument('--mean_pooled', type=int, default=0, choices=[0, 1],
                        help='If 1, mean-pool activations per protein for multi-layer mode')
    parser.add_argument('--layers_to_extract', type=int, nargs='+', default=None,
                        help='Specific layers to extract (if None and not bos_vs_meanpool, extracts all layers)')

    # Normalization
    parser.add_argument('--norm_scalars_path', type=str, default=None,
                        help='Path to load normalization scalars (required for normalized activations)')
    parser.add_argument('--compute_norm_scalars', type=int, default=0, choices=[0, 1],
                        help='If 1 and norm_scalars_path not found, compute new scalars')
    parser.add_argument('--norm_samples', type=int, default=500,
                        help='Number of proteins to sample for computing normalization scalars')

    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save extracted activations')
    parser.add_argument('--proteins_per_file', type=int, default=10000,
                        help='Number of proteins per output file (chunk size)')

    # System arguments
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='CUDA device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def load_protein_sequences(
    uniref_file: str,
    max_seq_len: int,
    num_proteins: int,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Load and randomly sample protein sequences from UniRef file.
    
    Returns:
        sequences: List of protein sequences
        metadata: List of sequence descriptions/IDs
    """
    print(f"Loading sequences from {uniref_file}...")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Target number of proteins: {num_proteins:,}")
    
    # First pass: collect all valid sequences
    valid_sequences = []
    valid_metadata = []
    
    total_processed = 0
    too_long = 0
    last_print_time = datetime.now()
    
    try:
        with gzip.open(uniref_file, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                total_processed += 1
                seq_str = str(record.seq)
                
                if len(seq_str) > max_seq_len:
                    too_long += 1
                    continue
                
                valid_sequences.append(seq_str)
                valid_metadata.append(record.description)
                
                # Progress update
                current_time = datetime.now()
                if (current_time - last_print_time).total_seconds() >= 10:
                    print(f"\rProcessed: {total_processed:,} | Valid: {len(valid_sequences):,} | Too long: {too_long:,}", end="")
                    sys.stdout.flush()
                    last_print_time = current_time
                
                # Stop if we have enough sequences to sample from
                # Collect 2x what we need for good random sampling
                if len(valid_sequences) >= num_proteins * 2:
                    print(f"\nCollected {len(valid_sequences):,} valid sequences for sampling")
                    break
    
    except Exception as e:
        print(f"\nError loading sequences: {e}")
        raise
    
    print(f"\n\nLoading complete:")
    print(f"  Total processed: {total_processed:,}")
    print(f"  Valid sequences: {len(valid_sequences):,}")
    print(f"  Too long (skipped): {too_long:,}")
    
    # Randomly sample the requested number of proteins
    if len(valid_sequences) < num_proteins:
        print(f"Warning: Only {len(valid_sequences)} valid sequences available, using all of them")
        selected_indices = list(range(len(valid_sequences)))
    else:
        random.seed(seed)
        selected_indices = random.sample(range(len(valid_sequences)), num_proteins)
    
    selected_sequences = [valid_sequences[i] for i in selected_indices]
    selected_metadata = [valid_metadata[i] for i in selected_indices]
    
    print(f"Selected {len(selected_sequences):,} proteins for extraction")
    
    return selected_sequences, selected_metadata


def extract_batch_activations_bos_vs_meanpool(
    sequences: List[str],
    metadata: List[str],
    esm_model,
    alphabet,
    device: torch.device,
    target_layer: int,
    normalization_scalars: Optional[Dict] = None
) -> torch.Tensor:
    """
    Extract BOS vs mean-pool activations for a batch of sequences.
    This matches EXACTLY what the collate function does during online training.
    
    Returns:
        activations: [batch_size, 2, d_model] tensor
    """
    batch_converter = alphabet.get_batch_converter()
    
    # Prepare batch data for ESM
    batch_data = [(meta, seq) for meta, seq in zip(metadata, sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
    
    # Single ESM forward pass for entire batch
    with torch.no_grad():
        results = esm_model(
            batch_tokens.to(device),
            repr_layers=[target_layer],
            return_contacts=False
        )
    
    bos_embeddings = []
    meanpool_embeddings = []
    
    for seq_idx in range(len(sequences)):
        actual_seq_len = len(sequences[seq_idx])
        
        # Get full representation including BOS token
        full_repr = results["representations"][target_layer][seq_idx]  # [seq_len_with_special, D]
        
        # Source 0: BOS token (position 0)
        bos_emb = full_repr[0]  # [D]
        
        # Source 1: Mean-pooled tokens (positions 1 to L, excluding BOS and EOS)
        token_emb = full_repr[1:actual_seq_len+1]  # [L, D]
        meanpool_emb = token_emb.mean(dim=0)  # [D]
        
        # Apply normalization if scalars are provided
        if normalization_scalars is not None:
            if 'bos' in normalization_scalars:
                bos_emb = bos_emb / normalization_scalars['bos']
            if 'meanpool' in normalization_scalars:
                meanpool_emb = meanpool_emb / normalization_scalars['meanpool']
        
        bos_embeddings.append(bos_emb)
        meanpool_embeddings.append(meanpool_emb)
    
    # Stack to create final batch tensor: [batch_size, 2, D]
    bos_stack = torch.stack(bos_embeddings, dim=0)  # [batch_size, D]
    meanpool_stack = torch.stack(meanpool_embeddings, dim=0)  # [batch_size, D]
    combined_batch = torch.stack([bos_stack, meanpool_stack], dim=1)  # [batch_size, 2, D]
    
    return combined_batch


def extract_batch_activations_multilayer(
    sequences: List[str],
    metadata: List[str],
    esm_model,
    alphabet,
    device: torch.device,
    layers_to_extract: List[int],
    normalization_scalars: Optional[Dict] = None,
    mean_pooled: bool = False
) -> torch.Tensor:
    """
    Extract multi-layer activations for a batch of sequences.
    This matches EXACTLY what the collate function does during online training.
    
    Returns:
        If mean_pooled: [batch_size, n_sources, d_model]
        Otherwise: [total_tokens_in_batch, n_sources, d_model]
    """
    batch_converter = alphabet.get_batch_converter()
    
    # Prepare batch data for ESM
    batch_data = [(meta, seq) for meta, seq in zip(metadata, sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
    
    # Single ESM forward pass for entire batch
    with torch.no_grad():
        results = esm_model(
            batch_tokens.to(device),
            repr_layers=layers_to_extract,
            return_contacts=False
        )
    
    n_sources = len(layers_to_extract)
    layer_token_collections = [[] for _ in range(n_sources)]
    
    # Extract embeddings for each sequence in the batch
    for seq_idx in range(len(sequences)):
        actual_seq_len = len(sequences[seq_idx])
        for layer_idx_pos, layer_idx in enumerate(layers_to_extract):
            # Extract embeddings, removing BOS and EOS tokens
            embedding = results["representations"][layer_idx][seq_idx, 1:actual_seq_len+1]  # [actual_len, D]
            
            if mean_pooled:
                # Reduce across tokens to a single vector per sequence
                pooled = embedding.mean(dim=0)  # [D]
                if normalization_scalars is not None and layer_idx in normalization_scalars:
                    scalar = normalization_scalars[layer_idx]
                    pooled = pooled / scalar
                layer_token_collections[layer_idx_pos].append(pooled)
            else:
                # Token-level path
                if normalization_scalars is not None and layer_idx in normalization_scalars:
                    scalar = normalization_scalars[layer_idx]
                    embedding = embedding / scalar
                layer_token_collections[layer_idx_pos].append(embedding)
    
    # Stack layers to create final batch tensor
    if mean_pooled:
        stacked_layers = []
        for layer_tokens in layer_token_collections:
            stacked_layer = torch.stack(layer_tokens, dim=0)  # [batch_size, D]
            stacked_layers.append(stacked_layer)
        combined_batch = torch.stack(stacked_layers, dim=1)  # [batch_size, n_sources, D]
    else:
        concatenated_layers = []
        for layer_tokens in layer_token_collections:
            concatenated_layer = torch.cat(layer_tokens, dim=0)  # [total_tokens, D]
            concatenated_layers.append(concatenated_layer)
        combined_batch = torch.stack(concatenated_layers, dim=1)  # [total_tokens, n_sources, D]
    
    return combined_batch


def save_chunk(
    activations: torch.Tensor,
    metadata: List[str],
    sequences: List[str],
    chunk_idx: int,
    output_dir: Path,
    config: Dict
):
    """Save a chunk of activations to disk."""
    chunk_data = {
        'activations': activations.cpu(),  # [num_proteins, n_sources, d_model]
        'metadata': metadata,
        'sequences': sequences,
        'chunk_idx': chunk_idx,
        'num_proteins': len(sequences),
    }
    
    filename = f"chunk_{chunk_idx:05d}.pt"
    filepath = output_dir / filename
    torch.save(chunk_data, filepath)
    
    return filepath


def main():
    args = parse_arguments()
    
    print("=" * 70)
    print("Offline Activation Extraction for Crosscoder Training")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"UniRef file: {args.uniref_file}")
    print(f"ESM model: {args.esm_model}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Number of proteins: {args.num_proteins:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"BOS vs mean-pool mode: {bool(args.bos_vs_meanpool)}")
    if args.bos_vs_meanpool:
        print(f"Target layer: {args.target_layer}")
    print(f"Output directory: {args.output_dir}")
    print(f"Proteins per file: {args.proteins_per_file:,}")
    print("=" * 70)
    
    # Set seeds
    set_all_seeds(args.seed)
    
    # Setup device
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ESM model
    print(f"\nLoading ESM model: {args.esm_model}")
    model_dict = {
        "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
    }
    esm_model, alphabet = model_dict[args.esm_model]()
    esm_model = esm_model.to(device)
    esm_model.eval()
    
    print(f"ESM model loaded: {esm_model.num_layers} layers, {esm_model.embed_dim} embedding dim")
    
    # Determine layers to extract
    if args.bos_vs_meanpool:
        target_layer = args.target_layer
        if target_layer is None:
            target_layer = esm_model.num_layers - 1
        layers_to_extract = [target_layer]
        n_sources = 2
        print(f"\nBOS vs mean-pool mode:")
        print(f"  Target layer: {target_layer}")
        print(f"  Source 0: BOS token")
        print(f"  Source 1: Mean-pooled tokens")
    else:
        if args.layers_to_extract is None:
            layers_to_extract = list(range(esm_model.num_layers + 1))
        else:
            layers_to_extract = args.layers_to_extract
        n_sources = len(layers_to_extract)
        print(f"\nMulti-layer mode:")
        print(f"  Layers: {layers_to_extract}")
        print(f"  Mean pooled: {bool(args.mean_pooled)}")
    
    # Load or compute normalization scalars
    normalization_scalars = None
    if args.norm_scalars_path:
        norm_path = Path(args.norm_scalars_path)
        if norm_path.exists():
            print(f"\nLoading normalization scalars from: {args.norm_scalars_path}")
            normalization_scalars = torch.load(args.norm_scalars_path)
            print(f"Loaded scalars: {normalization_scalars}")
        elif args.compute_norm_scalars:
            print(f"\nComputing normalization scalars...")
            norm_computer = LayerNormalizationComputer(
                esm_model, alphabet, device, layers_to_extract,
                bool(args.mean_pooled), bool(args.bos_vs_meanpool),
                target_layer if args.bos_vs_meanpool else None
            )
            normalization_scalars = norm_computer.compute_normalization_scalars(
                args.uniref_file, args.max_seq_len, args.norm_samples
            )
            # Save the computed scalars
            norm_computer.save_scalars(args.norm_scalars_path)
        else:
            print(f"Warning: Normalization scalars file not found: {args.norm_scalars_path}")
            print("Proceeding without normalization")
    else:
        print("\nNo normalization scalars path provided - proceeding without normalization")
    
    # Load protein sequences
    sequences, metadata = load_protein_sequences(
        args.uniref_file,
        args.max_seq_len,
        args.num_proteins,
        args.seed
    )
    
    # Configuration to save with the data
    config = {
        'esm_model': args.esm_model,
        'max_seq_len': args.max_seq_len,
        'num_proteins': len(sequences),
        'batch_size': args.batch_size,
        'bos_vs_meanpool': bool(args.bos_vs_meanpool),
        'target_layer': target_layer if args.bos_vs_meanpool else None,
        'mean_pooled': bool(args.mean_pooled),
        'layers_to_extract': layers_to_extract,
        'n_sources': n_sources,
        'd_model': esm_model.embed_dim,
        'normalization_scalars': normalization_scalars,
        'proteins_per_file': args.proteins_per_file,
        'seed': args.seed,
        'extraction_timestamp': datetime.now().isoformat(),
    }
    
    # Save config
    config_path = output_dir / "config.json"
    # Convert tensors/non-serializable to serializable format for JSON
    config_json = config.copy()
    if config_json['normalization_scalars'] is not None:
        config_json['normalization_scalars'] = {
            str(k): float(v) if isinstance(v, (int, float)) else v 
            for k, v in config_json['normalization_scalars'].items()
        }
    with open(config_path, 'w') as f:
        json.dump(config_json, f, indent=2)
    print(f"\nSaved config to: {config_path}")
    
    # Also save as .pt for easy loading
    torch.save(config, output_dir / "config.pt")
    
    # Extract activations in batches and save in chunks
    print(f"\nExtracting activations...")
    
    chunk_activations = []
    chunk_metadata = []
    chunk_sequences = []
    chunk_idx = 0
    total_extracted = 0
    
    # Process in batches
    num_batches = (len(sequences) + args.batch_size - 1) // args.batch_size
    
    for batch_start in tqdm(range(0, len(sequences), args.batch_size), total=num_batches, desc="Processing batches"):
        batch_end = min(batch_start + args.batch_size, len(sequences))
        batch_sequences = sequences[batch_start:batch_end]
        batch_metadata = metadata[batch_start:batch_end]
        
        # Extract activations for this batch
        if args.bos_vs_meanpool:
            batch_activations = extract_batch_activations_bos_vs_meanpool(
                batch_sequences,
                batch_metadata,
                esm_model,
                alphabet,
                device,
                target_layer,
                normalization_scalars
            )
        else:
            batch_activations = extract_batch_activations_multilayer(
                batch_sequences,
                batch_metadata,
                esm_model,
                alphabet,
                device,
                layers_to_extract,
                normalization_scalars,
                bool(args.mean_pooled)
            )
        
        # Add to current chunk
        chunk_activations.append(batch_activations.cpu())
        chunk_metadata.extend(batch_metadata)
        chunk_sequences.extend(batch_sequences)
        total_extracted += len(batch_sequences)
        
        # Check if we should save the current chunk
        if len(chunk_metadata) >= args.proteins_per_file:
            # Concatenate all activations in this chunk
            combined_activations = torch.cat(chunk_activations, dim=0)
            
            # Save chunk
            filepath = save_chunk(
                combined_activations,
                chunk_metadata,
                chunk_sequences,
                chunk_idx,
                output_dir,
                config
            )
            print(f"\nSaved chunk {chunk_idx}: {len(chunk_metadata)} proteins -> {filepath}")
            
            # Reset for next chunk
            chunk_activations = []
            chunk_metadata = []
            chunk_sequences = []
            chunk_idx += 1
    
    # Save any remaining activations
    if len(chunk_metadata) > 0:
        combined_activations = torch.cat(chunk_activations, dim=0)
        filepath = save_chunk(
            combined_activations,
            chunk_metadata,
            chunk_sequences,
            chunk_idx,
            output_dir,
            config
        )
        print(f"\nSaved final chunk {chunk_idx}: {len(chunk_metadata)} proteins -> {filepath}")
        chunk_idx += 1
    
    # Save manifest
    manifest = {
        'num_chunks': chunk_idx,
        'total_proteins': total_extracted,
        'config': config_json,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Extraction Complete!")
    print("=" * 70)
    print(f"Total proteins extracted: {total_extracted:,}")
    print(f"Number of chunks: {chunk_idx}")
    print(f"Activation shape per protein: [{n_sources}, {esm_model.embed_dim}]")
    print(f"Output directory: {output_dir}")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()


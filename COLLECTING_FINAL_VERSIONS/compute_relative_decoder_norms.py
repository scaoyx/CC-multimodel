#!/usr/bin/env python3
"""
Script to compute relative decoder norms and cosine similarities for each hidden neuron 
across all input sources/modalities from a saved TopK crosscoder checkpoint.

For each hidden neuron, computes:
- L2 norm of decoder weights for each input source/modality
- Relative norm = source_norm / sum_of_all_source_norms
- Cosine similarity between each source's decoder column and the dominant source's decoder column

Usage:
    python compute_relative_decoder_norms.py --checkpoint_path path/to/checkpoint.ckpt --output_path results.pt
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
import json
from topk_from_old_repo import LitLit


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compute relative decoder norms from saved checkpoint')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the saved checkpoint file (.ckpt)')
    parser.add_argument('--output_path', type=str, default='relative_decoder_norms.pt',
                        help='Path to save the computed relative norms')
    parser.add_argument('--output_format', type=str, choices=['pt', 'json', 'npz'], default='pt',
                        help='Output format (pt=PyTorch tensor, json=JSON, npz=NumPy)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to load the model on (cpu/cuda)')
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Load the trained model from checkpoint
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    print(f"Model hyperparameters: {hparams}")
    
    # Load the model using the class method
    model = LitLit.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Number of sources: {model.n_sources}")
    print(f"Hidden dimension: {model.hidden_dim}")
    print(f"Input dimensions: {[model.model.d_in] * model.n_sources}")
    
    return model


def compute_relative_decoder_norms(model):
    """
    Compute relative decoder norms for each hidden neuron across all sources/modalities
    
    Args:
        model: Trained LitLit model
        
    Returns:
        relative_norms: torch.Tensor of shape [hidden_dim, n_sources]
                       Each row sums to 1.0, representing the relative contribution
                       of each source to that hidden neuron
        absolute_norms: torch.Tensor of shape [hidden_dim, n_sources]
                       Raw L2 norms before normalization
        cosine_similarities: torch.Tensor of shape [hidden_dim, n_sources]
                           Cosine similarity between each source's decoder column
                           and the dominant source's decoder column for each neuron
    """
    print("Computing relative decoder norms and cosine similarities...")
    
    # Get decoder weights: [hidden_dim, n_sources, d_model]
    W_dec = model.model.W_dec.data
    hidden_dim, n_sources, d_model = W_dec.shape
    
    print(f"Decoder weight shape: {W_dec.shape}")
    
    # Compute L2 norm for each hidden neuron and each source
    # Shape: [hidden_dim, n_sources]
    absolute_norms = torch.norm(W_dec, dim=2, p=2)
    
    print(f"Absolute norms shape: {absolute_norms.shape}")
    print(f"Absolute norms stats - min: {absolute_norms.min():.6f}, max: {absolute_norms.max():.6f}, mean: {absolute_norms.mean():.6f}")
    
    # Compute sum of norms for each hidden neuron across all sources
    # Shape: [hidden_dim]
    total_norms = absolute_norms.sum(dim=1, keepdim=True)
    
    # Avoid division by zero (though this shouldn't happen in practice)
    total_norms = torch.clamp(total_norms, min=1e-12)
    
    # Compute relative norms: each row sums to 1.0
    # Shape: [hidden_dim, n_sources]
    relative_norms = absolute_norms / total_norms
    
    print(f"Relative norms shape: {relative_norms.shape}")
    print(f"Relative norms stats - min: {relative_norms.min():.6f}, max: {relative_norms.max():.6f}")
    
    # Verify that each row sums to approximately 1.0
    row_sums = relative_norms.sum(dim=1)
    print(f"Row sums stats - min: {row_sums.min():.6f}, max: {row_sums.max():.6f}, mean: {row_sums.mean():.6f}")
    
    # Check for any neurons with all-zero weights (shouldn't happen but good to check)
    zero_neurons = (total_norms.squeeze() < 1e-10).sum().item()
    if zero_neurons > 0:
        print(f"Warning: Found {zero_neurons} neurons with near-zero total decoder norm")
    
    # Compute cosine similarities with dominant source for each neuron
    print("Computing cosine similarities with dominant sources...")
    
    # Find the dominant source (highest norm) for each neuron
    # Shape: [hidden_dim]
    dominant_source_indices = absolute_norms.argmax(dim=1)
    
    # Initialize cosine similarities tensor
    # Shape: [hidden_dim, n_sources]
    cosine_similarities = torch.zeros_like(absolute_norms)
    
    # For each hidden neuron, compute cosine similarity with dominant source
    for neuron_idx in range(hidden_dim):
        dominant_source_idx = dominant_source_indices[neuron_idx].item()
        
        # Get the dominant source's decoder column for this neuron
        # Shape: [d_model]
        dominant_decoder = W_dec[neuron_idx, dominant_source_idx, :]
        
        # Compute cosine similarity between all sources and the dominant source
        for source_idx in range(n_sources):
            current_decoder = W_dec[neuron_idx, source_idx, :]
            
            # Compute cosine similarity
            cosine_sim = F.cosine_similarity(current_decoder.unsqueeze(0), 
                                           dominant_decoder.unsqueeze(0), 
                                           dim=1).item()
            cosine_similarities[neuron_idx, source_idx] = cosine_sim
    
    print(f"Cosine similarities shape: {cosine_similarities.shape}")
    print(f"Cosine similarities stats - min: {cosine_similarities.min():.6f}, max: {cosine_similarities.max():.6f}, mean: {cosine_similarities.mean():.6f}")
    
    # The diagonal should be 1.0 (similarity with itself)
    diagonal_similarities = cosine_similarities[torch.arange(hidden_dim), dominant_source_indices]
    print(f"Dominant source self-similarities - min: {diagonal_similarities.min():.6f}, max: {diagonal_similarities.max():.6f}")
    
    return relative_norms, absolute_norms, cosine_similarities


def save_results(relative_norms, absolute_norms, cosine_similarities, output_path, format='pt', model_info=None):
    """
    Save the computed norms and cosine similarities to file
    """
    print(f"Saving results to: {output_path}")
    
    if format == 'pt':
        results = {
            'relative_norms': relative_norms,
            'absolute_norms': absolute_norms,
            'cosine_similarities': cosine_similarities,
            'model_info': model_info
        }
        torch.save(results, output_path)
        
    elif format == 'json':
        results = {
            'relative_norms': relative_norms.tolist(),
            'absolute_norms': absolute_norms.tolist(),
            'cosine_similarities': cosine_similarities.tolist(),
            'model_info': model_info
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
    elif format == 'npz':
        np.savez(output_path, 
                relative_norms=relative_norms.numpy(),
                absolute_norms=absolute_norms.numpy(),
                cosine_similarities=cosine_similarities.numpy())
    
    print(f"Results saved successfully!")


def analyze_results(relative_norms, absolute_norms, cosine_similarities):
    """
    Print some basic analysis of the computed norms and cosine similarities
    """
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    
    hidden_dim, n_sources = relative_norms.shape
    
    print(f"Total hidden neurons: {hidden_dim}")
    print(f"Number of sources/modalities: {n_sources}")
    
    # Find neurons that are dominated by a single source
    max_relative_per_neuron, dominant_source = relative_norms.max(dim=1)
    
    print(f"\nDominant source analysis:")
    for source_idx in range(n_sources):
        count = (dominant_source == source_idx).sum().item()
        percentage = count / hidden_dim * 100
        print(f"  Source {source_idx}: {count} neurons ({percentage:.1f}%)")
    
    # Statistics on how specialized neurons are
    print(f"\nSpecialization statistics:")
    print(f"  Mean max relative norm: {max_relative_per_neuron.mean():.4f}")
    print(f"  Std max relative norm: {max_relative_per_neuron.std():.4f}")
    print(f"  Neurons with >90% weight in one source: {(max_relative_per_neuron > 0.9).sum().item()}")
    print(f"  Neurons with >75% weight in one source: {(max_relative_per_neuron > 0.75).sum().item()}")
    print(f"  Neurons with >50% weight in one source: {(max_relative_per_neuron > 0.5).sum().item()}")
    
    # Per-source statistics
    print(f"\nPer-source relative norm statistics:")
    for source_idx in range(n_sources):
        source_norms = relative_norms[:, source_idx]
        print(f"  Source {source_idx}: mean={source_norms.mean():.4f}, std={source_norms.std():.4f}")
    
    # Cosine similarity analysis
    print(f"\nCosine similarity analysis:")
    print(f"  Overall cosine similarity stats:")
    print(f"    Mean: {cosine_similarities.mean():.4f}")
    print(f"    Std: {cosine_similarities.std():.4f}")
    print(f"    Min: {cosine_similarities.min():.4f}")
    print(f"    Max: {cosine_similarities.max():.4f}")
    
    # Analyze cosine similarities for non-dominant sources
    print(f"\n  Non-dominant source cosine similarities:")
    for neuron_idx in range(min(hidden_dim, 10)):  # Show first 10 neurons as examples
        dominant_idx = dominant_source[neuron_idx].item()
        non_dominant_sims = []
        for source_idx in range(n_sources):
            if source_idx != dominant_idx:
                non_dominant_sims.append(cosine_similarities[neuron_idx, source_idx].item())
        if non_dominant_sims:
            print(f"    Neuron {neuron_idx} (dominant source {dominant_idx}): "
                  f"mean non-dominant similarity = {np.mean(non_dominant_sims):.4f}")
    
    # Count highly similar non-dominant sources
    similarity_threshold = 0.8
    high_similarity_count = 0
    for neuron_idx in range(hidden_dim):
        dominant_idx = dominant_source[neuron_idx].item()
        for source_idx in range(n_sources):
            if source_idx != dominant_idx and cosine_similarities[neuron_idx, source_idx] > similarity_threshold:
                high_similarity_count += 1
    
    total_non_dominant_pairs = hidden_dim * (n_sources - 1)
    high_sim_percentage = high_similarity_count / total_non_dominant_pairs * 100
    print(f"\n  Non-dominant sources with >={similarity_threshold} similarity: {high_similarity_count}/{total_non_dominant_pairs} ({high_sim_percentage:.2f}%)")


def main():
    args = parse_arguments()
    
    # Validate inputs
    if not Path(args.checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint_path, device=args.device)
    
    # Compute relative norms and cosine similarities
    relative_norms, absolute_norms, cosine_similarities = compute_relative_decoder_norms(model)
    
    # Analyze results
    analyze_results(relative_norms, absolute_norms, cosine_similarities)
    
    # Prepare model info for saving
    model_info = {
        'checkpoint_path': args.checkpoint_path,
        'hidden_dim': model.hidden_dim,
        'n_sources': model.n_sources,
        'd_model': model.model.d_in,
        'k': getattr(model.model, 'k', 'unknown')
    }
    
    # Save results
    save_results(relative_norms, absolute_norms, cosine_similarities, args.output_path, 
                args.output_format, model_info)
    
    print(f"\n{'='*50}")
    print("COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {args.output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main() 
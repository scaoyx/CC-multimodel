"""
Compute family specificity for crosscoder activations.

This script adapts interprot_compute_family_specificity.py to work with:
- Embeddings extracted from a crosscoder model (in .pt format)
- TSV files with 'Sequence' and 'Protein families' columns
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_embeddings_and_metadata(embeddings_path: Path):
    """
    Load embeddings from .pt file created by extract_protein_embeddings.py
    
    Returns:
        embeddings: torch.Tensor of shape [n_proteins, hidden_dim]
        metadata: dict with 'labels' and 'sequences'
    """
    print(f"Loading embeddings from {embeddings_path}")
    data = torch.load(embeddings_path, map_location='cpu')
    
    embeddings = data['embeddings']
    metadata = data.get('metadata', {})
    processing_info = data.get('processing_info', {})
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of proteins: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Processing info: {processing_info}")
    
    return embeddings, metadata


def load_tsv_with_families(tsv_path: Path, metadata: dict):
    """
    Load TSV file and match with metadata from embeddings.
    
    Args:
        tsv_path: Path to TSV file with 'Sequence' and 'Protein families' columns
        metadata: Metadata dict with 'labels' and 'sequences' from embeddings file
    
    Returns:
        pd.DataFrame with Entry, Sequence, and Protein_families columns
        np.ndarray: Valid indices (which proteins from embeddings to keep)
    """
    print(f"Loading TSV from {tsv_path}")
    df_tsv = pd.read_csv(tsv_path, sep='\t', low_memory=False)
    
    print(f"TSV columns: {df_tsv.columns.tolist()}")
    
    # Check required columns
    if 'Sequence' not in df_tsv.columns:
        raise ValueError("TSV must have 'Sequence' column")
    if 'Protein families' not in df_tsv.columns:
        raise ValueError("TSV must have 'Protein families' column")
    
    # Create a mapping from sequence to protein families
    seq_to_families = dict(zip(df_tsv['Sequence'].astype(str), 
                               df_tsv['Protein families'].fillna('')))
    
    # Match with metadata sequences
    if 'sequences' not in metadata or 'labels' not in metadata:
        raise ValueError("Metadata must contain 'sequences' and 'labels'")
    
    sequences = metadata['sequences']
    labels = metadata['labels']
    
    print(f"Number of sequences in embeddings: {len(sequences)}")
    
    # Build matched dataframe with indices
    matched_data = []
    for idx, (seq, label) in enumerate(zip(sequences, labels)):
        families = seq_to_families.get(seq, '')
        matched_data.append({
            'original_idx': idx,
            'Entry': label,
            'Sequence': seq,
            'Protein_families': families
        })
    
    df = pd.DataFrame(matched_data)
    
    # Parse protein families (split by semicolon)
    df['Protein_families'] = df['Protein_families'].astype(str).str.strip(';').str.split(';')
    
    # Remove entries with no families, but keep track of original indices
    valid_mask = df['Protein_families'].apply(lambda x: len(x) > 0 and x[0] != '')
    df_filtered = df[valid_mask].reset_index(drop=True)
    valid_indices = df_filtered['original_idx'].values
    
    print(f"Number of proteins with family annotations: {len(df_filtered)}")
    
    return df_filtered, valid_indices


def compute_family_specific_features(
    tsv_path: Path,
    embeddings_path: Path,
    class_list_col: str = "Protein_families",
):
    """
    Compute family specific features for crosscoder embeddings.
    
    This function:
    1. Loads embeddings from .pt file (shape: [n_proteins, hidden_dim])
    2. Loads TSV with protein families
    3. For each dimension in the embeddings, finds the protein with highest activation
    4. Optimizes classification threshold for each family of that protein
    
    Args:
        tsv_path: Path to TSV file with 'Sequence' and 'Protein families' columns
        embeddings_path: Path to .pt file with crosscoder embeddings
        class_list_col: Column name containing the list of protein families
    
    Returns:
        pd.DataFrame: Family specific features for each dimension
    """
    # Load embeddings and metadata
    embeddings, metadata = load_embeddings_and_metadata(embeddings_path)
    
    # Load TSV and match with metadata
    df, valid_indices = load_tsv_with_families(tsv_path, metadata)
    
    # Convert embeddings to numpy and filter to only valid proteins
    embeddings_np = embeddings.numpy()  # [n_proteins, hidden_dim]
    embeddings_filtered = embeddings_np[valid_indices]  # [n_valid_proteins, hidden_dim]
    
    # Transpose to [hidden_dim, n_proteins] format expected by the analysis
    activations = embeddings_filtered.T  # [hidden_dim, n_valid_proteins]
    total_dims = activations.shape[0]
    n_proteins = activations.shape[1]
    
    print(f"\nActivations shape: {activations.shape}")
    print(f"Total dimensions: {total_dims}")
    print(f"Total proteins (with families): {n_proteins}")
    
    # Verify dimensions match
    assert n_proteins == len(df), f"Mismatch: {n_proteins} proteins in activations but {len(df)} in dataframe"
    
    # Process each dimension
    all_results = []
    for dim in tqdm(range(total_dims), desc="Processing dimensions"):
        df_dim = add_normalized_acts(df.copy(), activations, dim)
        df_dim = df_dim.sort_values('act', ascending=False)
        
        # Get the highest activating protein
        first_row = df_dim[df_dim['act'] == 1.0].head(1)
        
        if len(first_row) == 0:
            continue
            
        highest_act_families = first_row[class_list_col].iloc[0]
        
        if highest_act_families and len(highest_act_families) > 0 and highest_act_families[0] != '':
            for family_name in highest_act_families:
                if not family_name or family_name.strip() == '':
                    continue
                    
                df_class = add_class_label(df_dim.copy(), class_list_col, family_name)
                
                # Skip if not enough positive or negative examples
                n_positive = df_class['class'].sum()
                n_negative = len(df_class) - n_positive
                
                if n_positive < 5 or n_negative < 5:
                    continue
                
                result = optimize_f1_boundary(df_class)
                result['dim'] = dim
                result['class'] = family_name
                all_results.append(result)
    
    if len(all_results) == 0:
        print("WARNING: No results generated!")
        return pd.DataFrame()
    
    return pd.DataFrame(all_results)


def optimize_f1_boundary(df: pd.DataFrame, x_col="act", y_col="class"):
    """
    Optimize the threshold for classifying a sequence as a member of the family.
    
    This function iterates through possible thresholds and selects the one that
    maximizes the F1 score on a training set, then evaluates on a test set.
    """
    # Extract features and labels
    X = df[x_col].values
    y = df[y_col].values
    
    # Split into test and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )
    
    best_f1 = 0
    best_threshold = 0
    
    # Iterate through possible thresholds
    for threshold in np.linspace(0.3, 0.9, 7):
        # Predictions based on threshold
        y_pred = (X_train >= threshold).astype(int)
        
        # Calculate F1 score
        current_f1 = f1_score(y_train, y_pred, zero_division=0)
        
        # Update best F1 and threshold if current is better
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
    
    # Evaluate on test set
    y_pred = (X_test >= best_threshold).astype(int)
    
    # Handle case where all predictions are the same class
    try:
        roc_auc = roc_auc_score(y_test, y_pred)
    except ValueError:
        roc_auc = 0.5
    
    return {
        "threshold": best_threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "n_data_points": int(y.sum()),
    }


def add_normalized_acts(df: pd.DataFrame, activations: np.ndarray, dim: int):
    """
    Add normalized activations for a specific dimension to the dataframe.
    
    Args:
        df: DataFrame with protein data
        activations: Array of shape [total_dims, n_proteins]
        dim: Dimension index to extract
    
    Returns:
        DataFrame with added 'act' column containing normalized activations
    """
    dim_activations = activations[dim]
    max_val = dim_activations.max()
    
    if max_val > 0:
        normalized_acts = dim_activations / max_val
    else:
        normalized_acts = dim_activations
    
    df['act'] = normalized_acts
    return df


def add_class_label(df: pd.DataFrame, col_name: str, class_name: str):
    """
    Add binary class label based on whether protein belongs to specified family.
    
    Args:
        df: DataFrame with protein data
        col_name: Column containing list of protein families
        class_name: Target family name
    
    Returns:
        DataFrame with added 'class' column (1 if protein in family, 0 otherwise)
    """
    # Explode the list column
    df_exploded = df.explode(col_name)
    
    # Add class label
    df_exploded['class'] = (df_exploded[col_name] == class_name).astype(int)
    
    # Remove duplicates, prioritizing rows with class=1
    df_exploded = df_exploded.sort_values('class', ascending=False)
    df_exploded = df_exploded.drop_duplicates(subset=['Entry'], keep='first')
    
    return df_exploded

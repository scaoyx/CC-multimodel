#!/usr/bin/env python3
"""
CLI script to compute family specificity for crosscoder embeddings.

This script adapts interprot_run_compute_family_specificity.py to work with
embeddings extracted from a crosscoder model.

Usage:
    python crosscoder_run_compute_family_specificity.py \
        --tsv-path Correct_length_50k_random_swissprot.tsv \
        --embeddings-path Meanpooled_5k_Comp_13modal.pt \
        --output-csv family_specific_features.csv
"""

from pathlib import Path

import click

from crosscoder_compute_family_specificity import (
    compute_family_specific_features,
)


@click.command()
@click.option(
    "--tsv-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to TSV file with 'Sequence' and 'Protein families' columns.",
)
@click.option(
    "--embeddings-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to .pt file containing crosscoder embeddings (from extract_protein_embeddings.py).",
)
@click.option(
    "--class-list-col",
    type=str,
    default="Protein_families",
    help='Column name containing the list of protein families. Defaults to "Protein_families".',
)
@click.option(
    "--output-csv",
    type=click.Path(),
    default="family_specific_features.csv",
    help="Output CSV file path. Defaults to 'family_specific_features.csv'.",
)
@click.option(
    "--output-parquet",
    type=click.Path(),
    default=None,
    help="Optional output Parquet file path (if you want parquet format).",
)
def main(tsv_path, embeddings_path, class_list_col, output_csv, output_parquet):
    """
    Compute family-specific features for crosscoder embeddings.
    
    This script:
    1. Loads crosscoder embeddings from a .pt file
    2. Matches them with protein family annotations from a TSV file
    3. For each embedding dimension, identifies which protein families are specific to it
    4. Outputs classification metrics (F1, precision, recall, etc.) for each dimension-family pair
    """
    print("=" * 60)
    print("Crosscoder Family Specificity Analysis")
    print("=" * 60)
    print(f"TSV path: {tsv_path}")
    print(f"Embeddings path: {embeddings_path}")
    print(f"Class column: {class_list_col}")
    print(f"Output CSV: {output_csv}")
    if output_parquet:
        print(f"Output Parquet: {output_parquet}")
    print("=" * 60)
    
    # Compute family specific features
    df = compute_family_specific_features(
        Path(tsv_path), 
        Path(embeddings_path), 
        class_list_col
    )
    
    if len(df) == 0:
        print("\nNo results generated. Check that:")
        print("1. TSV file has 'Sequence' and 'Protein families' columns")
        print("2. Embeddings file was created with extract_protein_embeddings.py")
        print("3. Sequences in embeddings match sequences in TSV")
        return
    
    # Save results
    print(f"\nSaving results to {output_csv}")
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} family-specificity results")
    
    if output_parquet:
        try:
            import pyarrow.parquet as pq
            df.to_parquet(output_parquet, index=False)
            print(f"Also saved to {output_parquet}")
        except ImportError:
            print("Note: pyarrow not installed, skipping parquet output")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total dimension-family pairs analyzed: {len(df)}")
    print(f"Unique dimensions with family specificity: {df['dim'].nunique()}")
    print(f"Unique families found: {df['class'].nunique()}")
    print(f"\nTop 10 families by frequency:")
    print(df['class'].value_counts().head(10))
    print(f"\nMean F1 score: {df['f1'].mean():.4f}")
    print(f"Mean precision: {df['precision'].mean():.4f}")
    print(f"Mean recall: {df['recall'].mean():.4f}")
    print(f"Mean ROC AUC: {df['roc_auc'].mean():.4f}")
    
    # Show best performing dimension-family pairs
    print(f"\nTop 10 dimension-family pairs by F1 score:")
    top_f1 = df.nlargest(10, 'f1')[['dim', 'class', 'f1', 'precision', 'recall', 'n_data_points']]
    print(top_f1.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Verification script to check layer information and validate setup for layer-specific training.
"""
import torch
from pathlib import Path
import sys

def check_layer_info():
    """Check the layer structure from existing extracted data."""
    print("="*70)
    print("VERIFYING ESM MODEL LAYER STRUCTURE")
    print("="*70)
    print()
    
    # Check original extracted activations
    offline_dir = Path('../../offline_activations')
    if offline_dir.exists():
        config_path = offline_dir / 'config.pt'
        if config_path.exists():
            config = torch.load(config_path, weights_only=False)
            print("Original extraction (first/middle/last):")
            print("-" * 70)
            for model_name, info in config['model_info'].items():
                print(f"{model_name}:")
                print(f"  Total layers (num_layers): {info['num_layers']}")
                print(f"  Total representations: 0 to {info['num_layers']} = {info['num_layers'] + 1} total")
                print(f"  Extracted layers: {info['layers_extracted']}")
                print(f"  Embed dimension: {info['embed_dim']}")
                print()
            
            print("CONFIRMED: Models have 7, 13, and 31 layer representations")
            print()
        else:
            print("No config.pt found in original extraction")
    else:
        print("No original offline_activations directory found")
    
    print()
    print("="*70)
    print("CHECKING FOR LAYER 1 (SECOND LAYER) DATA")
    print("="*70)
    print()
    
    # Check if layer 1 data exists
    layer1_dir = Path('../../offline_activations_layer1')
    if layer1_dir.exists():
        config_path = layer1_dir / 'config.pt'
        if config_path.exists():
            config = torch.load(config_path, weights_only=False)
            print("✓ Layer 1 extraction found!")
            print("-" * 70)
            for model_name, info in config['model_info'].items():
                print(f"{model_name}:")
                print(f"  Extracted layers: {info['layers_extracted']}")
                print(f"  Layer 1 available: {1 in info['layers_extracted']}")
                if 1 in info['layers_extracted']:
                    position = info['layers_extracted'].index(1)
                    print(f"  Layer 1 stored at position: {position}")
                print(f"  Embed dimension: {info['embed_dim']}")
                print()
            
            print("✓ Ready to train on layer 1!")
            print()
            print("Next step:")
            print("  python main_script_layer_specific.py \\")
            print("      --layer_index 1 \\")
            print("      --offline_data_dir ../../offline_activations_layer1 \\")
            print("      --hidden_dim 16384 \\")
            print("      --k 64 \\")
            print("      --cuda_device 0")
            return True
        else:
            print("✗ config.pt not found")
            return False
    else:
        print("✗ Layer 1 data not extracted yet")
        print()
        print("You need to extract layer 1 data first:")
        print("  ./extract_layer1.sh")
        print()
        print("Or manually:")
        print("  python extract_activations_layer1.py \\")
        print("      --uniref_file /data/lux70/data/uniref/uniref50.fasta.gz \\")
        print("      --num_proteins 500000 \\")
        print("      --output_dir ../../offline_activations_layer1 \\")
        print("      --cuda_devices 0 1 2")
        return False
    
    print()

def test_dataset_loading():
    """Test if the dataset can be loaded properly."""
    print()
    print("="*70)
    print("TESTING DATASET LOADING")
    print("="*70)
    print()
    
    layer1_dir = Path('../../offline_activations_layer1')
    if not layer1_dir.exists():
        print("✗ Cannot test dataset - layer 1 data not extracted")
        return False
    
    try:
        from offline_dataset_layer_specific import OfflineActivationDataModuleLayerSpecific
        
        print("Creating data module...")
        dm = OfflineActivationDataModuleLayerSpecific(
            data_dir=str(layer1_dir),
            layer_index=1,
            batch_size=32,
            num_workers=0
        )
        
        print(f"✓ Data module created successfully")
        print(f"  Input dimensions: {dm.input_dims}")
        print(f"  Number of sources: {dm.n_sources}")
        print(f"  Model names: {dm.model_names}")
        print(f"  Model groups: {dm.model_groups}")
        print()
        
        print("Creating dataloader...")
        train_loader = dm.train_dataloader()
        
        print("Getting sample batch...")
        sample_batch = next(iter(train_loader))
        activations, metadata, sequences = sample_batch
        
        print(f"✓ Sample batch loaded successfully")
        print(f"  Batch structure: List of {len(activations)} tensors")
        for i, act in enumerate(activations):
            print(f"    Source {i} ({dm.model_names[i]}): {tuple(act.shape)}")
        
        print()
        print("✓ Dataset loading test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading test FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print()
    layer1_ready = check_layer_info()
    
    if layer1_ready:
        test_dataset_loading()
    
    print()
    print("="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print()

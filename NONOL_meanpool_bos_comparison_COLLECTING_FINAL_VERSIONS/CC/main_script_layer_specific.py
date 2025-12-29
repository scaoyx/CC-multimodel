##FINAL REPRODUCIBLE
"""
Training script for layer-specific crosscoders.

Trains crosscoders for a specific layer index (e.g., layer 1, layer 2, etc.)
comparing the same layer across all three ESM models.

This is different from layerwise which only supports first/middle/last.
This script supports ANY layer index that was extracted.
"""
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch
import argparse
import multiprocessing
from datetime import datetime
import random
import numpy as np
from pathlib import Path


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)


class CheckpointLoggingCallback(pl.Callback):
    def __init__(self, wandb_logger):
        super().__init__()
        self.wandb_logger = wandb_logger

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint_path = trainer.checkpoint_callback.last_model_path
        
        if checkpoint_path and self.wandb_logger:
            self.wandb_logger.experiment.config.update(
                {"best_checkpoint_path": checkpoint_path},
                allow_val_change=True
            )
            self.wandb_logger.experiment.log({
                "checkpoint_path": checkpoint_path,
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step
            })
            print(f"\n{'='*80}")
            print(f"✓ Checkpoint saved: {checkpoint_path}")
            print(f"  Epoch: {trainer.current_epoch}, Step: {trainer.global_step}")
            print(f"{'='*80}\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train layer-specific crosscoders')

    # Layer selection
    parser.add_argument('--layer_index', type=int, default=1,
                        help='Specific layer index to train on (default: 1 = second layer)')

    # Data arguments
    parser.add_argument('--offline_data_dir', type=str, required=True,
                        help='Directory containing pre-extracted activations')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=16384,
                        help='Hidden dimension of the autoencoder')
    parser.add_argument('--k', type=int, default=64,
                        help='Number of top-k activations to keep')
    parser.add_argument('--encoder_decoder_init', type=int, default=1,
                        help='Whether to tie encoder-decoder weights (1=yes, 0=no)')

    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--inactive_threshold', type=int, default=200,
                        help='Threshold for considering neurons inactive')
    parser.add_argument('--aux_alpha', type=float, default=2,
                        help='Auxiliary loss weight')
    parser.add_argument('--k_aux', type=int, default=None,
                        help='Number of auxiliary neurons to use')
    parser.add_argument('--encoder_decoder_scale', type=float, default=None,
                        help='Scaling factor for encoder weights')
    parser.add_argument('--use_xavier_init', type=int, default=0, choices=[0, 1],
                        help='Use Xavier initialization for decoder weights')

    # System arguments
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='CUDA device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')

    # Validation cadence
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='How often to check validation (fraction of epoch)')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='Limit validation batches (fraction or number)')
    parser.add_argument('--active_neuron_check_interval', type=int, default=20,
                        help='Check active neurons every x batches')
    parser.add_argument('--wandb_key', type=str, default=None,
                        help='Wandb API key (optional if WANDB_API_KEY env var is set)')

    return parser.parse_args()


def main():
    if torch.cuda.is_available():
        multiprocessing.set_start_method('spawn', force=True)
    
    args = parse_arguments()
    
    # Validate arguments
    if not Path(args.offline_data_dir).exists():
        raise ValueError(f"Data directory does not exist: {args.offline_data_dir}")
    
    print("=" * 70)
    print(f"LAYER-SPECIFIC CROSSCODER TRAINING: LAYER {args.layer_index}")
    if args.layer_index == 1:
        print("Comparing SECOND LAYER across all three ESM models")
    print("=" * 70)
    print(f"Layer index: {args.layer_index}")
    print(f"Data directory: {args.offline_data_dir}")
    
    set_all_seeds(args.seed)
    
    # Create timestamp early
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create data module
    from offline_dataset_layer_specific import OfflineActivationDataModuleLayerSpecific
    
    data_module = OfflineActivationDataModuleLayerSpecific(
        data_dir=args.offline_data_dir,
        layer_index=args.layer_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=0.8,
        load_sequences=False,
        load_metadata=False,
        seed=args.seed,
    )
    
    # Get configuration
    n_sources = data_module.n_sources
    model_groups = data_module.model_groups
    model_names = data_module.model_names
    
    print(f"\nTraining {n_sources}-model layer-{args.layer_index} crosscoder")
    if args.layer_index == 1:
        print(f"Comparing second layer (index 1) across:")
        for i, name in enumerate(model_names):
            print(f"  - {name}: layer 1/{data_module.full_config['model_info'][name]['num_layers']} ({data_module.input_dims[i]}-dim)")
    else:
        print(f"Models: {', '.join(model_names)}")
    print(f"Model groups: {model_groups}")
    print(f"Per-model loss normalization: ENABLED")
    
    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    # Create save directory with layer-specific naming
    save_dir = Path("CC_layer_specific_checkpoints") / f"layer{args.layer_index}" / f"run_{timestamp}"
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create run name
    layer_desc = "SecondLayer" if args.layer_index == 1 else f"Layer{args.layer_index}"
    run_name = f"{layer_desc}_3models_h{args.hidden_dim}_k{args.k}_{timestamp}"
    
    wandb_logger = WandbLogger(
        project="CC-layer-specific-multimodel",
        name=run_name,
        save_dir=str(save_dir)
    )
    
    # Import model class
    from sparse_auto_script import LitLit
    
    # Get input dimensions from data module (actual dimensions per model)
    input_dims = data_module.input_dims
    print(f"Input dimensions: {input_dims}")
    
    # Create model
    model = LitLit(
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        k=args.k,
        encoder_decoder_init=args.encoder_decoder_init,
        learning_rate=args.learning_rate,
        inactive_threshold=args.inactive_threshold,
        aux_alpha=args.aux_alpha,
        active_neuron_check_interval=args.active_neuron_check_interval,
        k_aux=args.k_aux,
        encoder_decoder_scale=args.encoder_decoder_scale,
        use_xavier_init=bool(args.use_xavier_init),
        model_groups=model_groups,
        model_names=model_names,
    )
    
    print(f"\nCreated model:")
    print(f"  - Hidden dimension: {args.hidden_dim}")
    print(f"  - Input dimensions: {input_dims}")
    print(f"  - K (sparsity): {args.k}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Model groups: {model_groups}")
    print(f"  - Model names: {model_names}")
    
    # Save config
    config = {
        "layer_index": args.layer_index,
        "hidden_dim": args.hidden_dim,
        "k": args.k,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "n_sources": n_sources,
        "model_names": model_names,
        "model_groups": model_groups,
        "input_dims": input_dims,
        "seed": args.seed,
        "timestamp": timestamp,
    }
    
    import json
    with open(save_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nCheckpoints will be saved to: {checkpoint_dir}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='epoch={epoch:02d}-step={step}-val_loss={val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
        every_n_epochs=1,
        verbose=True
    )
    
    checkpoint_logging_callback = CheckpointLoggingCallback(wandb_logger)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, checkpoint_logging_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[int(args.cuda_device)] if torch.cuda.is_available() else 1,
        precision="16-mixed",
        log_every_n_steps=1,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
    )
    
    # Start training
    print("\n" + "=" * 70)
    print(f"STARTING TRAINING: LAYER {args.layer_index} CROSSCODER")
    print("=" * 70)
    
    trainer.fit(model, data_module)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"All checkpoints saved in: {checkpoint_dir}")


if __name__ == "__main__":
    main()

##FINAL REPRODUCIBLE
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
        if trainer.checkpoint_callback.best_model_path:
            self.wandb_logger.experiment.config.update({
                "best_checkpoint_path": trainer.checkpoint_callback.best_model_path
            })


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train N-modal sparse autoencoder (online or offline)')

    # Mode selection: online vs offline
    parser.add_argument('--offline_mode', type=int, default=0, choices=[0, 1],
                        help='If 1, use pre-extracted offline activations. If 0, use online ESM extraction.')
    parser.add_argument('--offline_data_dir', type=str, default=None,
                        help='Directory containing pre-extracted activations (required if offline_mode=1)')

    # Data arguments (for online mode)
    parser.add_argument('--uniref_file', type=str, default=None,
                        help='Path to UniRef FASTA file (required for online mode)')
    parser.add_argument('--esm_model', type=str, default="esm2_t6_8M_UR50D", 
                        choices=["esm2_t33_650M_UR50D", "esm2_t6_8M_UR50D", 
                                 "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D"],
                        help='ESM model to use for activation extraction (online mode only)')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--max_samples', type=int, default=50000000,
                        help='Maximum number of samples to use')
    parser.add_argument('--layers_to_extract', type=int, nargs='+', default=None,
                        help='Specific layers to extract (if None, extracts all layers)')
    
    # Layer normalization arguments (for online mode)
    parser.add_argument('--compute_layer_norm', type=int, default=1, choices=[0, 1],
                        help='Compute layer normalization scalars (1=yes, 0=no)')
    parser.add_argument('--norm_samples', type=int, default=500,
                        help='Number of proteins to sample for computing normalization scalars')
    parser.add_argument('--norm_scalars_path', type=str, default=None,
                        help='Path to save/load normalization scalars (optional)')

    # Model arguments
    parser.add_argument('--input_dims', type=int, nargs='+', required=False, default=None,
                        help='Input dimensions for each source (optional; inferred if not provided)')
    parser.add_argument('--hidden_dim', type=int, default=16384,
                        help='Hidden dimension of the autoencoder')
    parser.add_argument('--k', type=int, default=64,
                        help='Number of top-k activations to keep')
    parser.add_argument('--encoder_decoder_init', type=int, default=1,
                        help='Whether to tie encoder-decoder weights (1=yes, 0=no)')

    # JumpReLU parameters
    parser.add_argument('--use_jumprelu', type=int, default=1,
                        help='Whether to use JumpReLU activation (1=yes, 0=no)')
    parser.add_argument('--jump_eps', type=float, default=2.0,
                        help='JumpReLU epsilon parameter')
    parser.add_argument('--jump_c', type=float, default=4.0,
                        help='JumpReLU c parameter for sparsity scaling')
    parser.add_argument('--lambda_S', type=float, default=0.01,
                        help='Sparsity loss weight (lambda_S)')
    parser.add_argument('--lambda_P', type=float, default=3e-6,
                        help='Pre-activation penalty weight (lambda_P)')
    parser.add_argument('--t_init', type=float, default=-8,
                        help='Initial log-threshold value for JumpReLU')

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
                        help='Number of auxiliary neurons to use (default: min(hidden_dim // 2, 500))')
    parser.add_argument('--encoder_decoder_scale', type=float, default=None,
                        help='Scaling factor for encoder weights relative to decoder (default: d_in / 1000)')
    parser.add_argument('--use_xavier_init', type=int, default=0, choices=[0, 1],
                        help='Use Xavier initialization for decoder weights (1=yes, 0=no, default=0)')

    # System arguments
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='CUDA device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers (default: 4 for efficient parallel loading)')

    # Validation cadence
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='How often to check validation (fraction of epoch)')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='Limit validation batches (fraction or number)')
    parser.add_argument('--active_neuron_check_interval', type=int, default=20,
                        help='Check active neurons every x batches (default: 20)')
    parser.add_argument('--wandb_key', type=str, default=None,
                        help='Wandb API key (optional if WANDB_API_KEY env var is set)')

    # Mean pooling option
    parser.add_argument('--mean_pooled', type=int, default=0, choices=[0, 1],
                        help='If 1, mean-pool activations per protein -> [batch, n_sources, d_model]')
    
    # BOS vs mean-pool comparison mode
    parser.add_argument('--bos_vs_meanpool', type=int, default=0, choices=[0, 1],
                        help='If 1, use 2 sources (BOS token vs mean-pooled) from a single layer instead of comparing across layers')
    parser.add_argument('--target_layer', type=int, default=None,
                        help='Target layer to use for BOS vs mean-pool mode (default: last layer, e.g., 11 for esm2_t12)')
    
    # Resume training arguments
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--load_norm_scalars', type=str, default=None,
                        help='Path to .pt file with pre-computed layer normalization scalars (required when resuming)')

    return parser.parse_args()


def create_online_data_module(args, device):
    """Create the online data module that extracts ESM activations on-the-fly."""
    import esm
    from online_multilayer_datamodule import OnlineMultiLayerDataModule
    
    # Initialize ESM model
    print(f"Loading ESM model: {args.esm_model}")
    model_dict = {
        "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D
    }
    esm_model, alphabet = model_dict[args.esm_model]()
    esm_model = esm_model.to(device)
    esm_model.eval()
    
    print(f"ESM model has {esm_model.num_layers} layers")
    print(f"ESM embedding dimension: {esm_model.embed_dim}")

    # Handle normalization scalars for resume vs new training
    if args.resume_from_checkpoint is not None:
        # When resuming, load pre-computed scalars and disable computation
        print(f"Loading pre-computed normalization scalars from: {args.load_norm_scalars}")
        precomputed_scalars = torch.load(args.load_norm_scalars, weights_only=False)
        print(f"Loaded scalars for layers: {list(precomputed_scalars.keys())}")
        compute_layer_norm = False
        norm_scalars_path = args.load_norm_scalars
    else:
        compute_layer_norm = bool(args.compute_layer_norm)
        norm_scalars_path = args.norm_scalars_path
        precomputed_scalars = None

    # Create online data module
    data_module = OnlineMultiLayerDataModule(
        uniref_file=args.uniref_file,
        esm_model=esm_model,
        alphabet=alphabet,
        device=device,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        layers_to_extract=args.layers_to_extract,
        compute_layer_norm=compute_layer_norm,
        norm_samples=args.norm_samples,
        norm_scalars_path=norm_scalars_path,
        mean_pooled=bool(args.mean_pooled),
        bos_vs_meanpool=bool(args.bos_vs_meanpool),
        target_layer=args.target_layer
    )
    
    data_module.setup()
    
    # CRITICAL FIX: If resuming, manually set the normalization scalars AFTER setup()
    if args.resume_from_checkpoint is not None:
        data_module.normalization_scalars = precomputed_scalars
        underlying_dataset = data_module.train_dataset.dataset
        underlying_dataset.normalization_scalars = precomputed_scalars
        print(f"✓ Applied pre-computed normalization scalars to data module after setup")
        print(f"✓ Applied normalization scalars to underlying dataset")
    
    return data_module


def create_offline_data_module(args):
    """Create the offline data module that loads pre-extracted activations."""
    from offline_dataset import OfflineActivationDataModule
    
    print(f"Loading offline activations from: {args.offline_data_dir}")
    
    data_module = OfflineActivationDataModule(
        data_dir=args.offline_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=0.8,
        load_sequences=False,
        load_metadata=False,
        seed=args.seed,
    )
    
    # Offline data module already calls setup() in __init__
    # No ESM model needed!
    print(f"Offline mode: n_sources={data_module.n_sources}, d_model={data_module.d_model}")
    
    return data_module


def main():
    if torch.cuda.is_available():
        multiprocessing.set_start_method('spawn', force=True)
    args = parse_arguments()
    
    # Validate arguments based on mode
    if args.offline_mode:
        if args.offline_data_dir is None:
            raise ValueError("--offline_data_dir is required when --offline_mode=1")
        if not Path(args.offline_data_dir).exists():
            raise FileNotFoundError(f"Offline data directory not found: {args.offline_data_dir}")
        print("=" * 70)
        print("OFFLINE TRAINING MODE")
        print("=" * 70)
    else:
        if args.uniref_file is None:
            raise ValueError("--uniref_file is required when --offline_mode=0 (online mode)")
        print("=" * 70)
        print("ONLINE TRAINING MODE")
        print("=" * 70)
    
    # Validate resume arguments
    if args.resume_from_checkpoint is not None:
        if not Path(args.resume_from_checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_from_checkpoint}")
        if not args.offline_mode and args.load_norm_scalars is None:
            raise ValueError("When resuming from checkpoint in online mode, you must provide --load_norm_scalars path")
        if args.load_norm_scalars is not None and not Path(args.load_norm_scalars).exists():
            raise FileNotFoundError(f"Normalization scalars file not found: {args.load_norm_scalars}")
        print(f"RESUMING TRAINING from checkpoint: {args.resume_from_checkpoint}")
        if args.load_norm_scalars:
            print(f"Using pre-computed normalization scalars: {args.load_norm_scalars}")
    else:
        print("STARTING NEW TRAINING RUN")

    set_all_seeds(args.seed)

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create data module based on mode
    if args.offline_mode:
        data_module = create_offline_data_module(args)
    else:
        data_module = create_online_data_module(args, device)
    
    # Get n_sources from data module
    n_sources = data_module.n_sources
    print(f"Training {n_sources}-modal sparse autoencoder")
    
    # Get multi-model configuration if available
    model_groups = None
    model_names = None
    if hasattr(data_module, 'is_multimodel') and data_module.is_multimodel:
        model_groups = data_module.model_groups
        model_names = data_module.model_names
        print(f"Multi-model crosscoder configuration:")
        print(f"  Models: {', '.join(model_names)}")
        print(f"  Model groups: {model_groups}")
        print(f"  Each model has 3 concatenated layers (first, middle, last)")
        print(f"  Per-model loss normalization: ENABLED")
    
    if args.offline_mode:
        print(f"Using pre-extracted activations from: {args.offline_data_dir}")
        # Get mode info from offline config
        if hasattr(data_module, 'full_config') and data_module.full_config:
            config = data_module.full_config
            if config.get('bos_vs_meanpool'):
                print(f"BOS vs mean-pool mode: 2 sources from layer {config.get('target_layer')}")
                print(f"  Source 0: BOS token representation")
                print(f"  Source 1: Mean-pooled token representations")
    elif args.bos_vs_meanpool:
        print(f"BOS vs mean-pool mode: 2 sources from layer {data_module.target_layer}")
        print(f"  Source 0: BOS token representation")
        print(f"  Source 1: Mean-pooled token representations")
    else:
        print(f"Found {n_sources} layers to use as separate input modalities")

    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    # Create run name
    mode_str = "offline" if args.offline_mode else "online"
    resume_suffix = "_RESUMED" if args.resume_from_checkpoint else ""
    
    if args.offline_mode:
        run_name = f"{n_sources}Modal_SAE_{mode_str}_{random.randint(1000, 9999)}{resume_suffix}"
    else:
        run_name = f"{n_sources}Modal_SAE_{mode_str}_{args.esm_model}_{random.randint(1000, 9999)}{resume_suffix}"
    
    wandb_logger = WandbLogger(project="CC-concatenated-multimodel", name=run_name)

    sample_batch = next(iter(data_module.train_dataloader()))
    
    # Extract just the activations tensor from the batch tuple
    sample_activations = sample_batch[0]  # (activations, metadata, sequences)

    # Import model class
    from sparse_auto_script import LitLit

    # Handle model creation: resume from checkpoint or create new
    if args.resume_from_checkpoint is not None:
        # Load model from checkpoint
        print(f"Loading model from checkpoint: {args.resume_from_checkpoint}")
        model = LitLit.load_from_checkpoint(args.resume_from_checkpoint)
        print(f"Resumed model hyperparameters:")
        print(f"  - Hidden dimension: {model.hparams.hidden_dim}")
        print(f"  - Input dimensions: {model.hparams.input_dims}")
        print(f"  - K (sparsity): {model.hparams.k}")
        print(f"  - Learning rate: {model.hparams.learning_rate}")
        
        # Verify compatibility with current data
        expected_n_sources = len(model.hparams.input_dims)
        if n_sources != expected_n_sources:
            raise ValueError(
                f"Data module has {n_sources} sources but checkpoint expects {expected_n_sources} sources. "
                f"Make sure the data configuration matches the original training."
            )
    else:
        # Create new model
        # Infer input dimensions if not provided: [d_model] repeated n_sources
        if args.input_dims is None:
            args.input_dims = [sample_activations.shape[2]] * n_sources
            print(f"Inferred input dimensions: {args.input_dims}")
        elif len(args.input_dims) != n_sources:
            raise ValueError(f"Number of input_dims ({len(args.input_dims)}) must match number of activation sources ({n_sources})")

        print(f"Using input dimensions: {args.input_dims}")
        print(f"Sample batch shape: {tuple(sample_activations.shape)}")

        model = LitLit(
            input_dims=args.input_dims,
            hidden_dim=args.hidden_dim,
            k=args.k,
            encoder_decoder_init=args.encoder_decoder_init,
            learning_rate=args.learning_rate,
            inactive_threshold=args.inactive_threshold,
            aux_alpha=args.aux_alpha,
            active_neuron_check_interval=args.active_neuron_check_interval,
            use_jumprelu=bool(args.use_jumprelu),
            jump_eps=args.jump_eps,
            jump_c=args.jump_c,
            lambda_S=args.lambda_S,
            lambda_P=args.lambda_P,
            t_init=args.t_init,
            k_aux=args.k_aux,
            encoder_decoder_scale=args.encoder_decoder_scale,
            use_xavier_init=bool(args.use_xavier_init),
            model_groups=model_groups,
            model_names=model_names,
        )
        
        print(f"Created new model with:")
        print(f"  - Hidden dimension: {args.hidden_dim}")
        print(f"  - Input dimensions: {args.input_dims}")
        print(f"  - K (sparsity): {args.k}")
        print(f"  - Learning rate: {args.learning_rate}")
        if model_groups:
            print(f"  - Model groups: {model_groups}")
            print(f"  - Model names: {model_names}")

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='Topk_weights',
        filename=f'{n_sources}Modal_sae_{mode_str}_{start_time}_k{args.k}_hd{args.hidden_dim}_lr{args.learning_rate}_ep{args.max_epochs}',
        mode='min',
        save_top_k=1,
    )

    checkpoint_logging_callback = CheckpointLoggingCallback(wandb_logger)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, checkpoint_logging_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[int(args.cuda_device)] if torch.cuda.is_available() else None,
        precision="16-mixed",
        log_every_n_steps=1,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
    )

    # Start or resume training
    if args.resume_from_checkpoint is not None:
        trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model, data_module)


if __name__ == "__main__":
    main()

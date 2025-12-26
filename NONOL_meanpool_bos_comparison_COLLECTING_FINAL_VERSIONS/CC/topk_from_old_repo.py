#### identical to script for protein-level SAE

##FINAL REPRODUCIBLE
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import pytorch_lightning as pl
import wandb

import random
import numpy as np
import os

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seeds at the start of the script
set_seed()


class SparseActivation:
    #SparseActivation class following TopK class of Saefarer 
    def __init__(self, default_k, postact_fn=F.relu):
        self.default_k = default_k
        self.postact_fn = postact_fn # toggle as needed. If we want to use relu, else switch to identity 
        #should not make a big difference either way since model is incentivised to result in positive top activations anyway 
        self.srt = False #toggle as needed 

    def forward(self, features, k=None):
        num_active = k if k is not None else self.default_k
        top_vals, top_indexes = torch.topk(features, sorted=self.srt, k=num_active, dim=-1)
        activated = self.postact_fn(top_vals)
        output = torch.zeros_like(features)
        output.scatter_(-1, top_indexes, activated)
        return output, activated, top_indexes

class WeightTier:
    @staticmethod
    def forward(weights, input_dim, hidden_dim):
        temp_linear = nn.Linear(hidden_dim, input_dim, bias=False)
        print(f"temp_linear.weight.shape: {temp_linear.weight.shape}")
        weights[0].data = temp_linear.weight.clone()  
        print(f"self.weights[0].data.shape: {weights[0].data.shape}")
        weights[1].data = temp_linear.weight.T.clone() 
        print(f"self.weights[1].data.shape: {weights[1].data.shape}")
        del temp_linear
        print("Weights Tied successfully")
        return weights

class TopKAuto(nn.Module):
    def __init__(self, input_dims, hidden_dim, k, encoder_decoder_init, 
                 inactive_threshold=200, aux_alpha=0.03125, k_aux=None,
                 encoder_decoder_scale=None, use_xavier_init=False, model_groups=None):
        """
        Multi-modal TopK Autoencoder
        
        Args:
            input_dims: List of input dimensions for each source (e.g., [d_model, d_model, d_model])
            hidden_dim: Hidden dimension of the autoencoder
            k: Number of top-k activations to keep
            encoder_decoder_init: Whether to tie encoder-decoder weights
            inactive_threshold: Threshold for considering neurons inactive
            aux_alpha: Auxiliary loss weight
            k_aux: Number of auxiliary neurons to use (default: min(hidden_dim // 2, 500))
            encoder_decoder_scale: Scaling factor for encoder weights (default: d_in / 1000)
            use_xavier_init: Whether to use Xavier initialization for decoder weights
            model_groups: Optional list mapping each source to its model index for per-model loss normalization
                         e.g., [0, 1, 2] for 3 sources from 3 different models
        """
        super(TopKAuto, self).__init__()
        
        # Verify all input dimensions are the same (like CrossCoder)
        if len(set(input_dims)) != 1:
            raise ValueError(f"All input dimensions must be equal for multi-modal TopK. Got: {input_dims}")
        
        self.d_in = input_dims[0]  # All dimensions are the same
        self.n_sources = len(input_dims)
        self.hidden_dim = hidden_dim
        
        # Store model groups for per-model loss normalization
        # model_groups maps each source index to its model index
        # e.g., [0, 1, 2] means source 0 is from model 0, source 1 from model 1, etc.
        if model_groups is not None:
            self.register_buffer('model_groups', torch.tensor(model_groups, dtype=torch.long))
            self.n_models = len(set(model_groups))
        else:
            self.register_buffer('model_groups', None)
            self.n_models = self.n_sources
        
        # Initialize separate encoder weights for each source (like CrossCoder)
        self.W_enc = nn.Parameter(
            torch.empty(self.n_sources, self.d_in, hidden_dim)
        )
        
        # Initialize decoder weights (like CrossCoder)
        self.W_dec = nn.Parameter(torch.empty(hidden_dim, self.n_sources, self.d_in))
        if use_xavier_init:
            nn.init.xavier_uniform_(self.W_dec)
        else:
            bound = 1.0 / (self.d_in ** 0.5)
            self.W_dec.data.uniform_(-bound, bound)
        
        # Initialize encoder weights from decoder (like CrossCoder)
        # scaling_factor = self.d_in / hidden_dim
        if encoder_decoder_scale is not None:
            scaling_factor = encoder_decoder_scale
        else:
            scaling_factor = self.d_in / 1000  # Fixed scaling as if hidden_dim=1000
        with torch.no_grad():
            # Rearrange decoder weights to match encoder shape
            self.W_enc.data = scaling_factor * self.W_dec.data.permute(1, 2, 0)
        
        # Bias handling like CrossCoder: shared encoder bias, per-source decoder bias
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))
        self.b_dec = nn.Parameter(torch.zeros(self.n_sources, self.d_in))

        # Rest of initialization
        self.k = k
        self.inactive_threshold = inactive_threshold
        # self.k_aux = max(input_dims) // 2
        if k_aux is not None:
            self.k_aux = k_aux
        else:
            self.k_aux = min(hidden_dim // 2, 500)
        
        self.register_buffer('neuron_idle_counts', torch.zeros(hidden_dim, dtype=torch.long))
        
        self.sparse_activation = SparseActivation(default_k=self.k)
        self.aux_alpha = aux_alpha

    def encode(self, x):
        """
        Encode multi-modal input
        Args:
            x: [batch, n_sources, d_model] tensor
        Returns:
            combined latents before sparse activation
        """
        # Encode each source separately then sum (like CrossCoder)
        # x: [batch, n_sources, d_model]
        # W_enc: [n_sources, d_model, hidden_dim]
        
        # Compute encoded representations for each source
        encoded = torch.einsum('bsd,sdh->bh', x, self.W_enc)
        
        # Add shared encoder bias
        combined_latents = encoded + self.b_enc
        
        return combined_latents

    def decode(self, activations):
        """
        Decode activations to reconstruct all sources
        Args:
            activations: [batch, hidden_dim] sparse activations
        Returns:
            reconstructions: [batch, n_sources, d_model]
        """
        # activations: [batch, hidden_dim]
        # W_dec: [hidden_dim, n_sources, d_model]
        
        reconstructions = torch.einsum('bh,hsd->bsd', activations, self.W_dec)
        
        # Add per-source decoder bias
        reconstructions = reconstructions + self.b_dec.unsqueeze(0)
        
        return reconstructions

    def forward(self, x):
        """
        Forward pass for multi-modal input
        Args:
            x: [batch, n_sources, d_model] tensor containing all modalities
        """
        batch_size = x.shape[0]
        
        # Encode all sources
        combined_latents = self.encode(x)

        # Apply sparse activation on combined latents
        mid_sparse, activated, top_indexes = self.sparse_activation.forward(combined_latents)

        if self.training:
            self._update_neuron_activity_counts(mid_sparse)

        # Decode to reconstruct all sources
        reconstructions = self.decode(mid_sparse)

        # Calculate reconstruction loss for all sources
        main_loss = torch.mean((x - reconstructions) ** 2)

        # Calculate per-source losses for logging
        source_losses = []
        for i in range(self.n_sources):
            source_loss = torch.mean((x[:, i, :] - reconstructions[:, i, :]) ** 2)
            source_losses.append(source_loss)
        
        # Calculate per-model losses with normalization
        model_losses = []
        if self.model_groups is not None and len(self.model_groups) > 0:
            # Compute loss for each model separately, then average
            model_mse_sum = torch.zeros(self.n_models, device=x.device)
            model_counts = torch.zeros(self.n_models, device=x.device)
            
            for i in range(self.n_sources):
                model_idx = self.model_groups[i].item()
                source_mse = torch.mean((x[:, i, :] - reconstructions[:, i, :]) ** 2)
                model_mse_sum[int(model_idx)] += source_mse
                model_counts[int(model_idx)] += 1
            
            # Average MSE per model (normalizes for models with different numbers of sources)
            for model_idx in range(self.n_models):
                if model_counts[model_idx] > 0:
                    model_loss = model_mse_sum[model_idx] / model_counts[model_idx]
                    model_losses.append(model_loss)
                else:
                    model_losses.append(torch.tensor(0.0, device=x.device))

        aux_loss = torch.zeros((), device=x.device)
        inactive_mask = self.neuron_idle_counts > self.inactive_threshold

        if torch.count_nonzero(inactive_mask).item() != 0 and self.training:
            # Process inactive neurons
            aux_reconstructions = self._process_inactive_neurons(
                combined_latents, inactive_mask
            )

            scale = min(torch.count_nonzero(inactive_mask).item() / self.k_aux, 1.0)
            
            residual = x - reconstructions
            aux_residual = residual - aux_reconstructions
            
            aux_pre_alpha = torch.mean(aux_residual ** 2)
            aux_loss = self.aux_alpha * aux_pre_alpha

        return {
            'main_loss': main_loss,
            'aux_loss': aux_loss,
            'source_losses': source_losses,
            'model_losses': model_losses,
            'reconstructions': reconstructions
        }

    def _process_inactive_neurons(self, latents, inactive_mask):
        """Process inactive neurons for auxiliary loss"""
        masked_latents = latents.clone()
        masked_latents.data *= inactive_mask[None]
        
        inactive_activated_new, _, _ = self.sparse_activation.forward(
            masked_latents,
            min(torch.count_nonzero(inactive_mask).item(), self.k_aux)
        )

        aux_reconstructions = self.decode(inactive_activated_new)
        return aux_reconstructions

    def _update_neuron_activity_counts(self, activations):
        is_inactive = (activations == 0).all(dim=0).long()
        self.neuron_idle_counts *= is_inactive
        self.neuron_idle_counts += 1


#Lightning module for train and val 
class LitLit(pl.LightningModule):
    def __init__(self, input_dims, hidden_dim, k, encoder_decoder_init, 
                 learning_rate, inactive_threshold=200, aux_alpha=0.03125, 
                 active_neuron_check_interval=20, k_aux=None,
                 encoder_decoder_scale=None, use_xavier_init=False, 
                 model_groups=None, model_names=None, **kwargs):
        """
        Lightning module for multi-modal TopK autoencoder
        
        Args:
            input_dims: List of input dimensions for each source
            hidden_dim: Hidden dimension of the autoencoder
            k: Number of top-k activations to keep
            encoder_decoder_init: Whether to tie encoder-decoder weights (ignored, always tied)
            learning_rate: Learning rate for training
            inactive_threshold: Threshold for considering neurons inactive
            aux_alpha: Auxiliary loss weight
            active_neuron_check_interval: Check active neurons every x batches
            k_aux: Number of auxiliary neurons to use (default: min(hidden_dim // 2, 500))
            encoder_decoder_scale: Scaling factor for encoder weights (default: d_in / 1000)
            use_xavier_init: Whether to use Xavier initialization for decoder weights
            model_groups: Optional list mapping each source to its model index
            model_names: Optional list of model names for logging
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.hidden_dim = hidden_dim
        self.n_sources = len(input_dims)
        self.model_names = model_names
        
        self.model = TopKAuto(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            encoder_decoder_init=encoder_decoder_init,
            inactive_threshold=inactive_threshold,
            k=k,
            aux_alpha=aux_alpha,
            k_aux=k_aux,
            encoder_decoder_scale=encoder_decoder_scale,
            use_xavier_init=use_xavier_init,
            model_groups=model_groups
        )
        
        self.learning_rate = learning_rate
        
        # Active neuron tracking
        self.active_neuron_check_interval = active_neuron_check_interval
        self.batch_count = 0
        self.neuron_activity_tracker = torch.zeros(hidden_dim, dtype=torch.bool)
        self.last_active_check_batch = 0
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: [batch, n_sources, d_model] tensor
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Extract activations from batch tuple (activations, metadata, sequences)
        if isinstance(batch, tuple):
            activations = batch[0]  # [batch, n_sources, d_model]
        else:
            activations = batch  # backward compatibility
            
        outputs = self(activations)
        main_loss = outputs['main_loss']
        aux_loss = outputs['aux_loss']
        source_losses = outputs['source_losses']
        model_losses = outputs['model_losses']
        
        total_loss = main_loss + aux_loss
        
        # Track neuron activity for active neuron ratio metric
        with torch.no_grad():
            acts = self.model.encode(activations)
            sparse_acts, _, _ = self.model.sparse_activation.forward(acts)
            active_neurons_this_batch = (sparse_acts > 0).any(dim=0)  # [hidden_dim]
            self.neuron_activity_tracker = self.neuron_activity_tracker.to(active_neurons_this_batch.device)
            self.neuron_activity_tracker |= active_neurons_this_batch
            
            self.batch_count += 1
            
            # Check and log active neuron ratio every x batches
            if self.batch_count - self.last_active_check_batch >= self.active_neuron_check_interval:
                active_neuron_ratio = self.neuron_activity_tracker.float().mean().item()
                self.log('train_active_neuron_ratio', active_neuron_ratio, prog_bar=True, on_step=True)
                
                # Reset tracker for next interval
                self.neuron_activity_tracker.fill_(False)
                self.last_active_check_batch = self.batch_count
        
        # Log losses
        self.log('train_main_loss', main_loss, prog_bar=True, on_step=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=True)
        self.log('train_aux_loss', aux_loss, prog_bar=True, on_step=True)
        
        # Log per-source losses
        for i, source_loss in enumerate(source_losses):
            self.log(f'train_source_{i}_loss', source_loss, prog_bar=True, on_step=True)
        
        # Log per-model losses (normalized)
        for i, model_loss in enumerate(model_losses):
            model_name = self.model_names[i] if self.model_names and i < len(self.model_names) else f'model_{i}'
            self.log(f'train_{model_name}_loss', model_loss, prog_bar=True, on_step=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        # Extract activations from batch tuple (activations, metadata, sequences)
        if isinstance(batch, tuple):
            activations = batch[0]  # [batch, n_sources, d_model]
        else:
            activations = batch  # backward compatibility
            
        outputs = self(activations)
        main_loss = outputs['main_loss']
        source_losses = outputs['source_losses']
        model_losses = outputs['model_losses']
        
        inactive_mask = self.model.neuron_idle_counts > self.model.inactive_threshold
        dead_ratio = inactive_mask.float().mean()
        
        # Log val_loss for model checkpoint monitoring (use main_loss)
        self.log('val_loss', main_loss, on_step=False, on_epoch=True)
        self.log('val_main_loss', main_loss, on_step=True)
        self.log('val_dead_ratio', dead_ratio, on_step=True)
        
        # Log per-source validation losses
        for i, source_loss in enumerate(source_losses):
            self.log(f'val_source_{i}_loss', source_loss, on_step=True)
        
        # Log per-model validation losses (normalized)
        for i, model_loss in enumerate(model_losses):
            model_name = self.model_names[i] if self.model_names and i < len(self.model_names) else f'model_{i}'
            self.log(f'val_{model_name}_loss', model_loss, on_step=True)
        
        return main_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999)  # Default Adam betas
        )
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    @classmethod
    def load_from_checkpoint(cls, ckpt_file, *args, **kwargs):
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage, weights_only=False)
        state_dict = checkpoint['state_dict']
        
        # Extract hyperparameters from checkpoint
        hparams = checkpoint.get('hyper_parameters', {})
        
        # Get dimensions from the state dict
        if 'model.W_enc' in state_dict:  # New multi-modal format
            encoder_shape = state_dict['model.W_enc'].shape
            n_sources, d_in, hidden_dim = encoder_shape
            input_dims = [d_in] * n_sources  # All dimensions are the same
        else:  # Old format (backward compatibility)
            encoder_shape_text = state_dict['model.weights_text.0'].shape
            input_dim_text, hidden_dim = encoder_shape_text
            
            encoder_shape_image = state_dict['model.weights_image.0'].shape
            input_dim_image, _ = encoder_shape_image
            
            input_dims = [input_dim_text, input_dim_image]
        
        # Get k from hyperparameters - raise error if not found
        if 'k' not in hparams:
            raise ValueError("Could not find 'k' in saved hyperparameters. Cannot load model without knowing k value.")
        k = hparams['k']
        print(f"Loading model with k={k}")
        
        # Get other hyperparameters with defaults
        learning_rate = hparams.get('learning_rate', 0.001)
        encoder_decoder_init = hparams.get('encoder_decoder_init', 1)
        active_neuron_check_interval = hparams.get('active_neuron_check_interval', 20)
        
        # Create model instance with extracted parameters
        model = cls(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            k=k,
            encoder_decoder_init=encoder_decoder_init,
            learning_rate=learning_rate,
            active_neuron_check_interval=active_neuron_check_interval,
        )
        
        # Load the state dict
        model.load_state_dict(state_dict, strict=True)
        return model






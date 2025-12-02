#### identical to script for protein-level SAE

##FINAL REPRODUCIBLE
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import einops

import pytorch_lightning as pl
import wandb

import random
import numpy as np
import os

# Import the modified TopK model instead of CrossCoder
from topk_from_old_repo import TopKAuto, LitLit

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

# Export the LitLit class from topk_from_old_repo for backward compatibility
# This allows main_script_chunks.py to import LitLit from this file as before
__all__ = ['LitLit']





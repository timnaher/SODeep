#%%
import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from model import SOD_v1
from hydra.utils import instantiate
from omegaconf import OmegaConf


cfg = OmegaConf.load(model_config_path)

# Instantiate the model
model = instantiate(cfg.model)
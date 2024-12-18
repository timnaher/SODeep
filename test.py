#%%
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl



# Example Usage
import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profilers import PyTorchProfiler
from utils.loggers import LogLearningRateCallback

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from model_dev3 import TDS_1D_v1



@hydra.main(config_path="configs/models", config_name="SOD_v1", version_base=None)
def main(cfg):
    # Set up model, data, and logger
    model = TDS_1D_v1(input_length=150, num_classes=3, learning_rate=0.01, nhead=4, num_attention_layers=2,
                 dim_feedforward=128, dropout=0.1, resblock_config=[{'in_filters': 64, 'out_filters': 64, 'dilation': 1},
                                                                   {'in_filters': 64, 'out_filters': 64, 'dilation': 2},
                                                                   {'in_filters': 64, 'out_filters': 64, 'dilation': 4}],
                 pred_threshold=0.8, smoothness_weight=0.1, transition_penalty_weight=0.1, lr_decay_nstep=4,
                 lr_decay_factor=0.5, intermediate_dim=128)


    datamodule = instantiate(cfg.datamodule)

    logger = TensorBoardLogger("tb_logs", name="combined_tds_model")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{logger.log_dir}/checkpoints",
        filename="epoch-{epoch:02d}-val_loss-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=5, mode="min"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=30,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, LogLearningRateCallback()],
        log_every_n_steps=1,
        accelerator="cpu"
    )

    # Train
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()


#%%
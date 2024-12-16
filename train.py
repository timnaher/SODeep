# Standard Library Imports
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


@hydra.main(config_path="configs/models", config_name="SOD_v1", version_base=None)
def main(cfg):
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.datamodule)

    # Logger: Pass the entire Hydra configuration for hyperparameter tracking
    logger = TensorBoardLogger("tb_logs", name=cfg.name)
    logger.log_hyperparams(cfg)

    # Checkpoints and Early Stopping
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{logger.log_dir}/checkpoints",  #  logger's directory
        filename="{epoch:02d}-{val_loss:.2f}",  
        save_top_k=1,
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )

    # Trainer Configuration
    trainer = pl.Trainer(
        min_epochs=30,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            LogLearningRateCallback()],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accelerator="cpu"
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()

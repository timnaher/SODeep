import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.loggers import LogLearningRateCallback

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="defaults", version_base="1.2")
def main(cfg: DictConfig):
    # Ensure log directory exists
    os.makedirs(cfg.log_dir, exist_ok=True)

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir=cfg.log_dir,  # Central directory
        name=None,             # Avoid subfolders like 'default'
        version=None           # Avoid nested folders
    )
    logger.log_hyperparams(cfg)
    version_dir = logger.log_dir

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{version_dir}/checkpoints",  # Save checkpoints in version_x/checkpoints
        filename="{epoch:02d}-{val_loss:.2f}",  
        save_top_k=5,
        mode="min",
    )

    # Early Stopping
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=20, mode="min"
    )

    # Trainer
    trainer = pl.Trainer(
        gradient_clip_algorithm="norm",
        gradient_clip_val=2,
        min_epochs=cfg.trainer.min_epochs,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            LogLearningRateCallback()],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accelerator="cpu"
    )

    # Train the model
    trainer.fit(hydra.utils.instantiate(cfg.model), hydra.utils.instantiate(cfg.datamodule))

if __name__ == "__main__":
    main()

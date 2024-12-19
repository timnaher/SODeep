#%%
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.loggers import LogLearningRateCallback


@hydra.main(config_path="configs/models", config_name="SOD_trans-lin", version_base=None)
def main(cfg: DictConfig):
    # Instantiate the model
    model = hydra.utils.instantiate(cfg.model)
    print(model._get_left_context())



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

    #trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()

# %%

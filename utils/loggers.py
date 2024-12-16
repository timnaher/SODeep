import torch
import torch.nn as nn
import pytorch_lightning as pl

class LogLearningRateCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Get the optimizer
        optimizer = trainer.optimizers[0]  # Assuming one optimizer
        current_lr = optimizer.param_groups[0]['lr']  # Access learning rate
        trainer.logger.log_metrics({"learning_rate": current_lr}, step=trainer.global_step)

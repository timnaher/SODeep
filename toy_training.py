import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch import nn
import torch.nn.functional as F

class ToyDataset(TensorDataset):
    def __init__(self, size=1000):
        x = torch.rand(size, 10)  # 10 features
        y = (x.sum(dim=1) > 5).float()  # Binary labels based on sum of features
        super().__init__(x, y)

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def main():
    # Dataset and DataLoader
    print("Using PyTorch version:", torch.__version__)  

    dataset = ToyDataset(size=1000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model and Trainer
    model = SimpleModel()
    trainer = Trainer(max_epochs=5)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()

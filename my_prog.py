CHECKPOINT_FILE = "example.ckpt"

import torch
import torchvision
import datasets
import transforms as tf
import csv,os,glob
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import vision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from torch.optim import Adam


dataset = datasets.VideoLabelDataset(
    "output.csv",
    transform=torchvision.transforms.Compose([
        tf.VideoFilePathToTensor(max_len=10, fps=2, padding_mode='last'),tf.VideoGrayscale(),
        tf.VideoResize([128, 128]),
    ])
)
data_loader = torch.utils.data.DataLoader(dataset, batch_size = 2, shuffle = True)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # first and last layer have the same size as generated Tensor
        self.encoder = nn.Sequential(nn.Linear(128 * 128 * 10, 256), nn.ReLU(), nn.Linear(256, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, 128 * 128 * 10))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
autoencoder = LitAutoEncoder()
trainer = pl.Trainer(gpus=1)

trainer.fit(autoencoder,data_loader)
trainer.save_checkpoint(CHECKPOINT_FILE)


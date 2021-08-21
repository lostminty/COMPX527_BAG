CHECKPOINT_FILE = "example.ckpt"
SEED = 1
import torch
import torchvision
import datasets
import transforms as tf
import csv, os, glob, re, sys
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import vision, VisionDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from torch.optim import Adam




class DCSASS(VisionDataset):
    def __init__(self, directory, transform=None):
        super().__init__(directory)

        self.directory = directory
        self.transform = transform
        self.video_files = []
        self.labels = []
        self.positives = []

        regex = re.compile(r".*_x264")
        for file in glob.glob(directory + "/Labels/*.csv"):
            for row in csv.reader(open(file)):
                if not row:
                    continue

                file_base = regex.match(row[0])
                file_base = file_base.group(0) if file_base else None
                if not file_base:
                    continue

                if file_base[0:4] == "oadA":  # Correct a bad file path.
                    file_base = "R" + file_base
                    row[0] = "R" + row[0]

                file_path = f"{directory}/{row[1]}/{file_base}.mp4/{row[0]}.mp4"
                if not os.path.exists(file_path):
                    continue

                self.video_files.append(file_path)
                self.labels.append(row[1])
                self.positives.append(bool(int(row[2])) if row[2] else False)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        video = self.video_files[index]
        label = self.labels[index]
        if self.transform:
            video = self.transform(video)
        return video, label


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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(sys.argv[0], "<path to \"DCSASS Dataset\" ZIP extraction>")
        sys.exit(1)
        
    torch.manual_seed(SEED)
    dataset = DCSASS(
        sys.argv[1],
        transform=torchvision.transforms.Compose([
            tf.VideoFilePathToTensor(max_len=10, fps=2, padding_mode='last'),
            tf.VideoGrayscale(),
            tf.VideoResize([128, 128]),
        ])
    )
    file_count = len(dataset)
    lengths = [int(file_count *0.8),int(file_count * 0.2)]
    train,val = torch.utils.data.random_split(dataset, lengths)
    
    train_data_loader = torch.utils.data.DataLoader(train, batch_size = 2, shuffle = True)
    val_data_loader = torch.utils.data.DataLoader(val, batch_size = 2, shuffle = True)

    autoencoder = LitAutoEncoder()
    trainer = pl.Trainer(gpus=1)

    trainer.fit(autoencoder,train_dataloaders=train_data_loader,val_dataloaders=val_data_loader)
    trainer.save_checkpoint(CHECKPOINT_FILE)

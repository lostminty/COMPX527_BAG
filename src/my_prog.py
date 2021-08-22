import csv
import glob
import os
import pickle
import re
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import VisionDataset, vision

import datasets
import transforms as tf

CHECKPOINT_FILE = "example.ckpt"
SEED = 1
NUM_WORKERS = 8
DIM_SCALE = [128, 128]
LEN_SAMPLE = 10


class DCSASS(VisionDataset):
    def __init__(self, directory, transform, pickled_dirname="pickle_jar"):
        super().__init__(directory)

        self.directory = directory
        self.transform = transform
        self.video_files = []
        self.labels = []
        self.positives = []

        if not self.transform:
            raise ValueError(
                "A transformer is required for outputting tensors.")

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
                #self.positives.append(bool(int(row[2])) if row[2] else False)
                self.labels.append(row[1] if row[2] else "normal")

        self.unique_labels = np.unique(self.labels)
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(self.unique_labels)

        self.cur_label = np.array([0] * len(self.unique_labels))
        self.pickled_dirname = pickled_dirname
        if not os.path.exists(pickled_dirname):
            os.mkdir(pickled_dirname)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        label = self.labels[index]

        # Avoided unneeded conditional checks for performance reasons, as this
        # is where the bottleneck is.
        # Comment out the pickling block below if you want to disable pickling
        # and instead uncomment the directly below line.
        # video = self.transform(self.video_files[index])

        pickled = f"{self.pickled_dirname}/{index}"
        if os.path.exists(pickled):
            video = pickle.load(open(pickled, "rb"))
        else:
            video = self.transform(self.video_files[index])
            pickle.dump(video, open(pickled, "wb"))

        label_index = self.encoder.transform([label])[0]
        return video, label_formatter(label_index, len(self.unique_labels))


def label_formatter(label, num_of_classes):
    label_encoded = [0] * num_of_classes
    label_encoded[label] += 1
    return torch.from_numpy(np.array(label_encoded))


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # first and last layer have the same size as generated Tensor
        self.encoder = nn.Sequential(nn.Linear(
            DIM_SCALE[0] * DIM_SCALE[1] * LEN_SAMPLE, 64), nn.ReLU(), nn.Linear(64, 14))
        self.decoder = nn.Sequential(nn.Linear(14, 64), nn.ReLU(), nn.Linear(
            64, DIM_SCALE[0] * DIM_SCALE[1] * LEN_SAMPLE))

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

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(x.size(0), -1)

        y_hat = self.encoder(x)

        loss = F.mse_loss(y_hat, y)

        self.log('val_loss', loss, prog_bar=False)
        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        # print(type(y_true))
        y_pred = label_formatter(y_pred[0], 14)
        # print(y_pred)
        return {'loss': loss,
                'y_true': y_true,
                'y_pred': y_pred}

    def validation_epoch_end(self, outputs):
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            y_true = np.append(y_true, results_dict['y_true'])
            y_pred = np.append(y_pred, results_dict['y_pred'])

        # print(np.shape(y_true))

        acc = accuracy_score(y_true, y_pred)
        self.log('val_acc', acc)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(sys.argv[0], "<path to \"DCSASS Dataset\" ZIP extraction>")
        sys.exit(1)

    torch.manual_seed(SEED)

    #label_encoder = torchvision.transforms.Lambda(lambda y,y_set: preprocessing.LabelEncoder().fit(y_set).transform([y]))

    dataset = DCSASS(
        sys.argv[1],
        transform=torchvision.transforms.Compose([
            tf.VideoFilePathToTensor(
                max_len=LEN_SAMPLE, fps=2, padding_mode='last'),
            tf.VideoGrayscale(),
            tf.VideoResize(DIM_SCALE),
        ])
    )

    file_count = len(dataset)
    lengths = [int(file_count * 0.8), int(file_count * 0.2)]

    diff = file_count - sum(lengths)

    if diff != 0:
        lengths[1] = lengths[1]+diff

    train, val = torch.utils.data.random_split(dataset, lengths)

    train_data_loader = torch.utils.data.DataLoader(
        train, batch_size=1, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_data_loader = torch.utils.data.DataLoader(
        val, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    autoencoder = LitAutoEncoder()
    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=3, verbose=False, mode="max")
    trainer = pl.Trainer(gpus=1, logger=tb_logger,
                         callbacks=[early_stop_callback])

    trainer.fit(autoencoder, train_dataloaders=train_data_loader,
                val_dataloaders=val_data_loader)
    trainer.save_checkpoint(CHECKPOINT_FILE)

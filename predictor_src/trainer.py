import numpy as np
import pytorch_lightning as pl
import torch
from litautoencoder import LitAutoEncoder
from dcsass_dataloader import DCSASS

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import datasets as datasets
import transforms as tf
LEN_SAMPLE=10
CHECKPOINT_FILE = "example.ckpt"
SEED = 1
NUM_WORKERS = 8
DIM_SCALE = [128,128]
TRANSFORMS = transforms.Compose([
            tf.VideoFilePathToTensor(
                max_len=LEN_SAMPLE, fps=2, padding_mode='last'),
            tf.VideoGrayscale(),
            tf.VideoResize(DIM_SCALE)])



def dataSplit(dataset,size_of_training_set):
    file_count = len(dataset)
    lengths = [int(file_count * size_of_training_set), int(file_count * (1.0-size_of_training_set))]
    diff = file_count - sum(lengths)
    if diff != 0:
        lengths[1] = lengths[1]+diff
    return torch.utils.data.random_split(dataset, lengths)
    

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(sys.argv[0], "<path to \"DCSASS Dataset\" ZIP extraction>")
        sys.exit(1)

    torch.manual_seed(SEED)
    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    train, val = dataSplit(dataset,0.8)
    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=3, verbose=False, mode="max")

    train_data_loader = torch.utils.data.DataLoader(
        train, batch_size=1, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_data_loader = torch.utils.data.DataLoader(
        val, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    autoencoder = LitAutoEncoder()
    trainer = pl.Trainer(gpus=-1,auto_select_gpus=True, logger=tb_logger,
                         callbacks=[early_stop_callback])

    trainer.fit(autoencoder,train_dataloaders=train_data_loader,
                val_dataloaders=val_data_loader)
    
    trainer.save_checkpoint(CHECKPOINT_FILE)

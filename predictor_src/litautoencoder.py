import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score


class LitAutoEncoder(pl.LightningModule):
    
    
    def __init__(self):
        super().__init__()
        
        self.dim_scale = [128,128]
        self.len_sample = 10
        # first and last layer have the same size as generated Tensor
        self.encoder = nn.Sequential(nn.Linear(
            self.dim_scale[0] * self.dim_scale[1] * self.len_sample, 64), nn.ReLU(), nn.Linear(64, 14))
        self.decoder = nn.Sequential(nn.Linear(14, 64), nn.ReLU(), nn.Linear(
            64, self.dim_scale[0] * self.dim_scale[1] * self.len_sample))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x,y_true = x
       
        x = x.view(x.size(0), -1)
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
        y_pred = self.label_formatter(y_pred[0], 14)
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

        acc = accuracy_score(y_true, y_pred)
        self.log('val_acc', acc)
        
        
    def label_formatter(self,label, num_of_classes):
        label_encoded = [0] * num_of_classes
        label_encoded[label] += 1
        return torch.from_numpy(np.array(label_encoded))




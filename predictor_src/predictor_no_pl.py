import torch
import json
from torch import nn
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score


# NOTE: This does not work. This class has no `predict()`.
class LitAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.dim_scale = [128, 128]
        self.len_sample = 10
        # first and last layer have the same size as generated Tensor
        self.encoder = nn.Sequential(nn.Linear(
            self.dim_scale[0] * self.dim_scale[1] * self.len_sample, 64), nn.ReLU(), nn.Linear(64, 14))
        self.decoder = nn.Sequential(nn.Linear(14, 64), nn.ReLU(), nn.Linear(
            64, self.dim_scale[0] * self.dim_scale[1] * self.len_sample))

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path), strict=False)
        self.eval()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x, _ = x

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

        acc = accuracy_score(y_true, y_pred)
        self.log('val_acc', acc)


LABELS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
          'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
          'Stealing', 'Vandalism', '-']


class loader(torch.utils.data.Dataset):
    def __init__(self, json_strings):
        self.video_files, self.labels = zip(
            *list(map(lambda x: json.loads(x), json_strings)))
        self.video_files = list(
            map(lambda x: torch.Tensor(x), self.video_files))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        # print(type(self.video_files[index]))
        return self.video_files[index], self.labels[index]


def predictor(json_strings, model_path="example.ckpt"):
    encoder = LitAutoEncoder()
    encoder.load(model_path)
    # trainer = pl.Trainer(gpus=-1, auto_select_gpus=True)
    # trainer = pl.Trainer(gpus=None)
    is_single = None

    if isinstance(json_strings, str):
        json_strings = [json_strings]
        is_single = True
    else:
        is_single = False
    test_dataset = loader(json_strings)

    data_loader = torch.utils.data.DataLoader(
        # test_dataset, batch_size=1, shuffle=True, pin_memory=True)
        test_dataset, batch_size=1, shuffle=True)

    # predictions = trainer.predict(encoder, data_loader)
    predictions = encoder.predict(data_loader)

    if is_single:
        return_val = torch.Tensor.tolist(predictions[0])
        return_val = {'label': LABELS[return_val.index(
            max(return_val))], 'confidence': max(return_val)}
    else:
        vals = list(map(lambda x: torch.Tensor.tolist(x), predictions))
        top_vals = [max(x) for x in vals]
        top_vals_indicies = [val_entry.index(
            x) for x, val_entry in zip(top_vals, vals)]
        return_val = [{"label": LABELS[index], "confidence":round(
            top_val[index], 3)} for index, top_val in zip(top_vals_indicies, top_vals)]

    return json.dumps(return_val)


def lambda_handler(event, context):
    return predictor(event)

from litautoencoder import LitAutoEncoder
import pytorch_lightning as pl
import torch
import json

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

        print(type(self.video_files[index]))
        return self.video_files[index], self.labels[index]


def predictor(json_strings, model_path="model.ckpt"):
    encoder = LitAutoEncoder()
    encoder.load_from_checkpoint(model_path)
    trainer = pl.Trainer(gpus=None, logger=False)
    is_single = None

    if isinstance(json_strings, str):
        json_strings = [json_strings]
        is_single = True
    else:
        is_single = False
    test_dataset = loader(json_strings)

    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True)

    predictions = trainer.predict(encoder, data_loader)

    if is_single:
        return_val = torch.Tensor.tolist(predictions[0])
        return_val = {'label':labels[return_val.index(max(return_val))],
                      'confidence':"{:.2%}".format(max(return_val))}
    else:
        vals = list(map(lambda x: torch.Tensor.tolist(x), predictions))
        top_vals = [max(x) for x in vals]
        top_vals_indicies = [val_entry.index(
            x) for x, val_entry in zip(top_vals, vals)]
        return_val = [{"label":LABELS[index],"confidence":"{:.2%}".format(top_val[index])} 
                      for index,top_val in zip(top_vals_indicies,top_vals)]

    return return_val  # Lambda instance will stringify it for us.


def lambda_handler(event, context):
    return predictor(event)

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
    torch.manual_seed(0)
    encoder = LitAutoEncoder()
    encoder.load_from_checkpoint(model_path)
    trainer = pl.Trainer(gpus=None, logger=False)

    
        
    test_dataset = loader(json_strings)

    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True)

    predictions = trainer.predict(encoder, data_loader)

    
    vals = torch.Tensor.tolist(predictions[0])[0]
    top_val =max(vals)
    index = vals.index(top_val)
    return_val = {"label":LABELS[index],"confidence":top_val}

    return return_val  # Lambda instance will stringify it for us.


def lambda_handler(event, context):
    return predictor(event)

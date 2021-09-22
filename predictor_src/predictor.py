from litautoencoder import LitAutoEncoder
import pytorch_lightning as pl
import torch
import json

MODEL_PATH = "model.ckpt"
LABELS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
          'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
          'Stealing', 'Vandalism', '-']

# Put everything that can be loaded just once up here. Have got 6x performance
# improvements from doing this and without memory leaks.
torch.manual_seed(0)
encoder = LitAutoEncoder()
encoder.load_from_checkpoint(MODEL_PATH)
trainer = pl.Trainer(gpus=None, logger=False, progress_bar_refresh_rate=0,
                     weights_summary=None)


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


# This function should be kept as quick as possible.
def lambda_handler(event, context):
    test_dataset = loader([event])  # `event` is the JSON strings.

    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    predictions = trainer.predict(encoder, data_loader)

    vals = torch.Tensor.tolist(predictions[0])[0]
    top_val = max(vals)
    index = vals.index(top_val)
    return_val = {"label": LABELS[index], "confidence": top_val}

    return return_val  # Lambda instance will stringify it for us.

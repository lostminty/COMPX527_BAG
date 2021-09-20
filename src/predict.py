import my_prog as mp
import pytorch_lightning as pl
import torch
import json


class loader(torch.utils.data.Dataset):
    def __init__(self, json_strings):
        self.video_files,self.labels = zip(*list(map(lambda x: json.loads(x),json_strings)))
        self.video_files = list(map(lambda x: torch.Tensor(x),self.video_files))
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, index):
        
        
        print(type(self.video_files[index]))
        return self.video_files[index],self.labels[index]
    


def predictor(json_strings,model_path="./model.pt"):
    encoder = mp.LitAutoEncoder()
    encoder.load_from_checkpoint(model_path)
    trainer = pl.Trainer(gpus=-1, auto_select_gpus=True)
    is_single = None
    
    if isinstance(json_strings, str):
        json_strings=[json_strings]
        is_single = True
    else:
        is_single = False
    test_dataset = loader(json_strings)
    
    
    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, pin_memory=True)
    
    predictions = trainer.predict(encoder,data_loader)
    
    
    if is_single:
        return_val = torch.Tensor.tolist(predictions[0])
        return_val = {'label':mp.labels[return_val.index(max(return_val))],'confidence':max(return_val)}
    else:
        vals=list(map(lambda x: torch.Tensor.tolist(x),predictions))
        top_vals = [max(x) for x in vals]
        top_vals_indicies =[val_entry.index(x) for x,val_entry in zip(top_vals,vals)]
        return_val = [{"label":mp.labels[index],"confidence":top_val[index]} for index,top_val in zip(top_vals_indicies,top_vals)]
    
    return json.dumps(return_val)
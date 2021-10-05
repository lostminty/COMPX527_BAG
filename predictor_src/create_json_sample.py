import numpy as np
import dcsass_dataloader
import json,sys
import transforms as tf
from torchvision import transforms

LEN_SAMPLE=10
DIM_SCALE = [128,128]
TRANSFORMS = transforms.Compose([
            tf.VideoFilePathToTensor(
                max_len=LEN_SAMPLE, fps=2, padding_mode='last'),
            tf.VideoGrayscale(),
            tf.VideoResize(DIM_SCALE)])

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print(sys.argv[0], "<path to \"DCSASS Dataset\" ZIP extraction> <name_of_output_file> <index of item (default=0)>")
        sys.exit(1)
    
    if len(sys.argv) == 4:
        index=int(sys.argv[3])
    else:
        index=0
        
    
    dataset=dcsass_dataloader.DCSASS(sys.argv[1],TRANSFORMS)
    
    item=dataset.__getitem__(index)
    
    item_list = [item[0].tolist(),item[1].tolist()]
    
    
    with open (sys.argv[2], "w") as f:
        json.dump(item_list, f)

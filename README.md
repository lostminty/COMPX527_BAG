# COMPX527_BAG (Behaviour Anomaly Group)
COMPX527-21B A2:AWS CC App Service to detect suspicious behaviour on submitted CCTV footage


Training on:
- https://www.kaggle.com/mateohervas/dcsass-dataset (requires sign in)
Has 13 classes and flags indicating a clip is of interest to the class or normal. so, 14 classes. could in theory look at it as 26 classes

No  te:
for downloading the Dataset, the zip comes with a duplicate of the dataset.


Based on:
- https://github.com/PyTorchLightning/pytorch-lightning
(pytorch helper module, automates a lot of 'boilerplate code'. Has some capacity to make code more portable: TODO!)
- https://github.com/YuxinZhaozyx/pytorch-VideoDataset 
(videodataset helper module, just uses the dataset.py and transforms.py. Expects them to be in the same dir: using only transforms.py)

most of what is in the [my_prog.py](my_prog.py) is from the README.md


# TODO
- portability



Some dependencies

- pytorch https://pytorch.org/ (will give you a conda command to run if you select appropriate for you system)
- pytorch lightning https://github.com/PyTorchLightning/pytorch-lightning
- CUDA toolkit https://developer.nvidia.com/cuda-downloads

Been running it in an anaconda env
https://www.anaconda.com/products/individual

after installing, reload command prompt/bash (windows has a shortcut installed in the start menu for anaconda)

in [my_prog.py](my_prog.py) are two vars:
- SEED = 1
- NUM_WORKERS = 8 (set to your number of CPU cores)

if you have multiple GPUs, edit the following line param gpus=x

[line to edit](my_prog.py#L120)
```python
trainer = pl.Trainer(gpus=1)
```


### Setup Instructions - FIX ###
1. Install nvidia cuda toolkit from link above
 
2. 
```sh
conda env create --file ./envs/video.yml
```
3. 
```sh
python my_prog.py /path/to/dcsass/videos/
```



## Notes ##

### Transforms ###
- VideoToTensor(frames_in_tensor=20,fps=2)
- Grayscale
- Resize(128,128)

### Label Encoder ###
- Not well implemented: creates a sklearn labelencoder within the dataset class init then uses a helper method to create numpy arrays
  
### Encoder ###
- Linear(in=video.x*video.y*frames_in_tensor,out=64)
- ReLU
- Linear(out=64, num_of_classes)

### Decoder ###
- Linear(in=num_of_classes,out=64)
- ReLU
- Linear(in=64,out=video.x*video.y*frames_in_tensor)

### Early Stop Callback ###
- generic callback, uses log_acc var. currently running tests

## TODO ##
- Training Step description
- Optimizer description
- validation step & epoch_end desc


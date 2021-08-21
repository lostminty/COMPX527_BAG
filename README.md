# COMPX527_BAG (Behaviour Anomaly Group)
COMPX527-21B A2:AWS CC App Service to detect suspicious behaviour on submitted CCTV footage


Training on:
- https://www.kaggle.com/mateohervas/dcsass-dataset

Note:
for downloading the Dataset, the zip comes with a duplicate of the dataset.
also,
to process correctly, move the Labels folder out of the main dataset directory (note where this is for using the csv helper script)

Based on:
- https://github.com/PyTorchLightning/pytorch-lightning
(pytorch helper module, automates a lot of 'boilerplate code'. Has some capacity to make code more portable: TODO!)
- https://github.com/YuxinZhaozyx/pytorch-VideoDataset 
(videodataset helper module, just uses the dataset.py and transforms.py. Expects them to be in the same dir)

most of what is in the [my_prog.py](my_prog.py) is from the README.md

modifications (from pytorch-lightening README.md demo code) are:
- using the datasets.py helper methods for making the dataset objects
- adjusting the Tensor parameters
- adjusting the resizing
- added Grayscale transform
- Added in gpus=1 parameter to trainer

TODO:
for the DataLoader, one can set the num_workers parameter but it's currently left at default (None) which might be a way to improve efficient utilising of system resources.



Some dependencies

- pytorch https://pytorch.org/ (will give you a conda command to run if you select appropriate for you system)
- pytorch lightning https://github.com/PyTorchLightning/pytorch-lightning
- CUDA toolkit https://developer.nvidia.com/cuda-downloads

Been running it in an anaconda env
https://www.anaconda.com/products/individual

after installing, reload command prompt/bash (windows has a shortcut installed in the start menu for anaconda)
```sh
conda create --name video
conda activate video
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge pytorch-lightning
conda install -c anaconda cudatoolkit
```

edit the vars in CAPS of [label_csv_helper.py](label_csv_helper.py) to point to your dataset, labels and CSV output (for use in [pytorch-VideoDataset](https://github.com/YuxinZhaozyx/pytorch-VideoDataset) class)
```sh
python label_csv_helper.py
```
edit my_prog.py to know where you put the CSV output from the previous step

```sh
python my_prog.py
```

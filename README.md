# COMPX527_BIG
COMPX527-21B A2:AWS CC App Service to detect suspicious behaviour on submitted CCTV footage


Training on:
https://www.kaggle.com/mateohervas/dcsass-dataset

Note:
for downloading the Dataset, the zip comes with a duplicate of the dataset.
also,
to process correctly, move the Labels folder out of the main dataset directory (note where this is for using the csv helper script)

Based on:
https://github.com/PyTorchLightning/pytorch-lightning
(pytorch helper module, automates a lot of 'boilerplate code'. Has some capacity to make code more portable: TODO!)
https://github.com/YuxinZhaozyx/pytorch-VideoDataset 
(videodataset helper module, just uses the dataset.py and transforms.py. Expects them to be in the same dir)

most of what is in the my_prog.py is from the README.md
modifications are:
using the datasets.py helper methods for making the dataset objects
adjusting the Tensor parameters
adjusting the resizing
added Grayscale transform
Added in gpus=1 parameter to trainer

TODO:
for the DataLoader, one can set the num_workers parameter but it's currently left at default (None) which might be a way to improve efficient utilising of system resources.

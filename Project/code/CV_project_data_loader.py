import torch
import torchfile
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

bus_path = 'C:/Users/nimro/Desktop/cv_project_local/busesTrain'
dictonary = {'a': 1, 'b':2}

class BusLoader(Dataset):
    def __init__(self, bus_folder, annotation_dict):
        self.fullpath = bus_folder
        self.busdata = os.listdir(bus_folder)
        self.annotations_dict = annotation_dict

    def __len__(self):
        return len(self.busdata)

    def __getitem__(self, idx):
        sample_name = self.busdata[idx]
        full_sample_path = os.path.join(self.fullpath, sample_name)
        sample = Image.open(full_sample_path)
        sample_labels = self.annotations_dict[sample_name]

        transform = transforms.Compose(transforms.ToTensor())
        sample_tensor = transform(sample)

        return sample_tensor.float(), torch.tensor(sample_labels)

import torch
# import torchfile
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

'''''################################# Locations ########################### '''''

File = 'C:/Users/Idan/Python_Proj/CV\Proj/annotationsTrain.txt'
bus_path = 'C:/Users/Idan/PycharmProjects/CV/busesTrain'

# bus_path = 'C:/Users/nimro/Desktop/cv_project_local/busesTrain'

'''''################################# Read text file to dictionary ########################### '''''

def read_txt(file):

    dictionary = {}
    with open(file) as f:
        for line in f:
            key, val = line.split(':')
            val_list = process_val(val)
            dictionary[key] = val_list
    return dictionary


def process_val(val):
    loc_list = []
    split = val.split(']')
    for word in split:

        if word == '\n':
            continue

        if word[0] == '[':
            word = word[1:]
        else:
            word = word[2:]
        word_split = word.split(',')
        word_split = list(map(int, word_split))
        word_np = np.asarray(word_split)
        loc_list.append(word_np)
    return loc_list


dictionary = read_txt(File)

'''''################################# Make data loader ########################### '''''


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
        sample = io.imread(full_sample_path)
        sample_labels = self.annotations_dict[sample_name]

        # transform = transforms.Compose(transforms.ToTensor())
        # sample_tensor = transform(sample)

        return sample, sample_labels


Loader = BusLoader(bus_path, dictionary)

'''''################################# Show example ########################### '''''


def show_example(loader, idx):

    img, labels = loader[idx]
    for label in labels:
        cv2.rectangle(img, (label[0], label[1]), (label[0]+label[2], label[1]+label[3]), (255, 0, 0), 3, cv2.LINE_AA)

    plt.figure()
    plt.imshow(img)
    plt.show()


show_example(Loader, 0)

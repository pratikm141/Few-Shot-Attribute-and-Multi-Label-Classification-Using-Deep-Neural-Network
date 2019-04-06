from __future__ import print_function
from collections import defaultdict
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import multiprocessing.pool
multiprocessing.cpu_count()
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import torchvision.models as models
import random
import os

print("preprocessing- converting data to tensor")
import pickle
import tqdm

file = open("./celeba/list_attr_celeba.txt", "r") 
i =0
row = []
for line in tqdm.tqdm(file):
    
    i=i+1
    if(i<3):
        continue
    linesplit = line.split(' ')
    linesplitsub = linesplit[1:]
    im = []
    for num in linesplitsub:
        if(num==''):
            continue
        im.append(float(num))
    row.append(im)

file.close()


labels = torch.Tensor(row)


pickle.dump(labels,open('./celeba/labelcelebA.pkl','wb'))

print("preprocessing done")

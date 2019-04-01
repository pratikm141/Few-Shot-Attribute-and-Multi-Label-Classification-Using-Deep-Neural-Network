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


facedatatrain =datasets.ImageFolder('./lfw/lfw-deepfunneled_reduced_bkup',
                                    transforms.Compose([transforms.Scale(64),transforms.ToTensor()]))

feat = []
import tqdm
for (x,y) in tqdm.tqdm(facedatatrain):
    feat.append(x)

feattensor = torch.stack(feat)

import pickle
pickle.dump(feat,open('./lfw/tensorFeat.pkl','wb'))


file = open("./lfw/lfw_attributes.txt", "r") 
i =0
row = []
for line in tqdm.tqdm(file):
    
    i=i+1
    if(i<3):
        continue
    linesplit = line.split('\t')
    linesplitsub = linesplit[2:]
    im = []
    for num in linesplitsub:
        im.append(float(num))
    row.append(im)

file.close()

labels = torch.Tensor(row)

boollabels = labels >0

floatlabels = boollabels.type(torch.FloatTensor)

pickle.dump(floatlabels,open('./lfw/tensorLabel.pkl','wb'))

print("preprocessing done")

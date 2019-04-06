from __future__ import print_function
from collections import defaultdict
import numpy as np
import torch
print("preprocessing- converting data to tensor")
import tqdm
labelfile = open("./birds/birds-train.arff","r")
featlist = []
labellist = []
cnt = 0
for line in tqdm.tqdm(labelfile):
    cnt = cnt + 1
    if(cnt<284):
        continue

    linelist = line.split(",")
    linefeat = linelist[:260]
    linelabel = linelist[260:]

    featlistrow = []
    labellistrow = []
    
    f=0
    for cell in linefeat:
        featlistrow.append(float(cell))
        
    for cell in linelabel:
        labellistrow.append(int(cell))
    
    featlist.append(featlistrow)
    labellist.append(labellistrow)

labelfile.close()

import tqdm
labelfile = open("./birds/birds-test.arff","r")
cnt = 0
for line in tqdm.tqdm(labelfile):
    cnt = cnt + 1
    if(cnt<284):
        continue

    linelist = line.split(",")
    linefeat = linelist[:260]
    linelabel = linelist[260:]

    featlistrow = []
    labellistrow = []
    
    f=0
    for cell in linefeat:
        featlistrow.append(float(cell))
        
    for cell in linelabel:
        labellistrow.append(int(cell))
    
    featlist.append(featlistrow)
    labellist.append(labellistrow)

labelfile.close()


arrfeat = np.array( featlist,dtype='float32' )
arrlabel = np.array( labellist,dtype='float32' )


tensorFeat = torch.from_numpy(arrfeat)
tensorLabel = torch.from_numpy(arrlabel)


tensorLabel = tensorLabel.type(torch.LongTensor)


import pickle
pickle.dump(tensorFeat,open("./birds/birdsFeat.pkl","wb"))
pickle.dump(tensorLabel,open("./birds/birdsLabel.pkl","wb"))

print("preprocessing done")

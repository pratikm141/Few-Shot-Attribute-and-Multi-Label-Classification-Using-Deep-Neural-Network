from __future__ import print_function
from collections import defaultdict
import numpy as np
import torch

import tqdm
labelfile =  open("../../data/slashdot/new/SLASHDOT-F.arff","r")
featlist = []
labellist = []
cnt = 0
for line in tqdm.tqdm(labelfile):
    rowtensor = torch.zeros(1101)
    
    cnt = cnt + 1
    if(cnt<1107):
        continue

    line = line[1:len(line)-2]

    linelist = line.split(",")
    
    
    for cell in linelist:
        comp = cell.split(" ")
        rowtensor[int(comp[0])] = int(comp[1])
    
    featlist.append(rowtensor[22:])
    labellist.append(rowtensor[:22])

labelfile.close()

tensorFeat = torch.stack(featlist)
tensorLabel = torch.stack(labellist)


tensorLabel = tensorLabel.type(torch.LongTensor)


import pickle
pickle.dump(tensorFeat,open("./slashdot/slashdotFeat.pkl","wb"))
pickle.dump(tensorLabel,open("./slashdot/slashdotLabel.pkl","wb"))

print("preprocessing done")

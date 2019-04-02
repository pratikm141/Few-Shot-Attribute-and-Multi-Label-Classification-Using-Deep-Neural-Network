from __future__ import print_function
from collections import defaultdict
import numpy as np
import torch

import tqdm
labelfile = open("../../data/llog/LLOG-F.arff","r")
featlist = []
labellist = []
cnt = 0
for line in tqdm.tqdm(labelfile):
    rowtensor = torch.zeros(1079)
    
    cnt = cnt + 1
    if(cnt<1085):
        continue

    line = line[1:len(line)-2]

    linelist = line.split(",")
    
    
    for cell in linelist:
        comp = cell.split(" ")
        rowtensor[int(comp[0])] = int(comp[1])
   
    
    featlist.append(rowtensor[75:])
    labellist.append(rowtensor[:75])

labelfile.close()

tensorFeat = torch.stack(featlist)
tensorLabel = torch.stack(labellist)


tensorLabel = tensorLabel.type(torch.LongTensor)


import pickle
pickle.dump(tensorFeat,open("./llog/llogFeat.pkl","wb"))
pickle.dump(tensorLabel,open("./llog/llogLabel.pkl","wb"))

print("preprocessing done")

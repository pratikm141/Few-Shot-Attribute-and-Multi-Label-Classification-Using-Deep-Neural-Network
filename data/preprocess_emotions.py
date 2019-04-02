from __future__ import print_function
from collections import defaultdict
import numpy as np
import torch

import tqdm
labelfile = open("./emotions/emotions.dat","r")
featlist = []
labellist = []
cnt = 0
for line in tqdm.tqdm(labelfile):
    cnt = cnt + 1
    if(cnt<83):
        continue
#     print('here')
    linelist = line.split(",")
    linefeat = linelist[:72]
    linelabel = linelist[72:]
#     print('line',linelist)
    featlistrow = []
    labellistrow = []
    
    f=0
    for cell in linefeat:
        featlistrow.append(float(cell))
        
    for cell in linelabel:
        labellistrow.append(int(cell))
    
    featlist.append(featlistrow)
    labellist.append(labellistrow)
#     break
labelfile.close()



arrfeat = np.array( featlist,dtype='float32' )
arrlabel = np.array( labellist,dtype='float32' )


tensorFeat = torch.from_numpy(arrfeat)
tensorLabel = torch.from_numpy(arrlabel)


tensorLabel = tensorLabel.type(torch.LongTensor)


import pickle
pickle.dump(tensorFeat,open("./emotions/emotionsFeat.pkl","wb"))
pickle.dump(tensorLabel,open("./emotions/emotionsLabel.pkl","wb"))

print("preprocessing done")

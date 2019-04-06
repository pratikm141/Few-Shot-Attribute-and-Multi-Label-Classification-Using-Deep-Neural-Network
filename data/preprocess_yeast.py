from __future__ import print_function
from collections import defaultdict
import numpy as np
import torch

print("preprocessing- converting data to tensor")

import tqdm
labelfile = open("./yeast/yeast.arff","r")
i = 0
feat = []
label = []
for line in labelfile:
    i = i+1
    if(i>=122):
        tmp =line[:len(line)-1].split(',')
        feat.append(tmp[:117-14])
        label.append(tmp[117-14:])

labelfile.close()

arrlabel = np.array( label,dtype='float32' )
arrfeat = np.array( feat,dtype='float32' )


tensorFeat = torch.from_numpy(arrfeat)
tensorLabel = torch.from_numpy(arrlabel)


tensorLabel = tensorLabel.type(torch.LongTensor)


import pickle
pickle.dump(tensorFeat,open("./yeast/yeastFeat.pkl","wb"))
pickle.dump(tensorLabel,open("./yeast/yeastLabel.pkl","wb"))

print("preprocessing done")

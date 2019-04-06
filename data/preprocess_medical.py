from __future__ import print_function
from collections import defaultdict
import numpy as np
import torch
print("preprocessing- converting data to tensor")
import tqdm
labelfile = open("./medical/medical.dat","r")
featlist = []
labellist = []
cnt = 0
for line in tqdm.tqdm(labelfile):
    cnt = cnt + 1
    if(cnt<1499):
        continue

    linelist = line.split(",")
    linefeat = linelist[:1449]
    linelabel = linelist[1449:]

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
pickle.dump(tensorFeat,open("./medical/medicalFeat.pkl","wb"))
pickle.dump(tensorLabel,open("./medical/medicalLabel.pkl","wb"))

print("preprocessing done")

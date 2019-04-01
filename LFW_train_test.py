
# coding: utf-8

# In[1]:


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



import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('float', metavar='N', type=int, nargs='+',help='an integer for the accumulator')
parser.add_argument('--no_cuda',default=False, type=bool)
parser.add_argument('--gpu',default=0, type=int)
parser.add_argument('--epochs',default=20, type=int)
parser.add_argument('--shot',default=20,type=int)
parser.add_argument('--train_way',default=5,type=int)
args = parser.parse_args()

if(args.epochs<2):
    print("--epochs should be more than 2")
    os._exit(1)

print('gpu ',args.gpu,' shot=',args.shot)

idgpu = args.gpu
usecuda = not args.no_cuda and torch.cuda.is_available()
batch_size = 64
epochs = args.epochs
seed = 1
loginterval= 200
total_labels= 73
torch.manual_seed(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(idgpu)
idgpu = 0
if usecuda:
    torch.cuda.manual_seed(seed)
    
kwargs = {}#{'num_workers': 1, 'pin_memory': True} if usecuda else {}


roundthreshold = 0.5

directory = './results/lfw/FSML{}shot'.format(args.shot)
if not os.path.exists(directory):
    os.makedirs(directory)




import pickle
labelcelebA = pickle.load(open('./data/lfw/tensorLabel.pkl','rb'))
labelcelebAzeroed = labelcelebA




import tqdm
attrListPos = []
attrListNeg = []
for i in tqdm.tqdm(range(total_labels)):
    attrPos = []
    attrNeg = []
    for j in range(len(labelcelebA[:,i])):
        if(labelcelebA[j,i]>0):
            attrPos.append(j)
        else:
            attrNeg.append(j)

    attrListPos.append(attrPos)
    attrListNeg.append(attrNeg)




facedatatrain = pickle.load(open('./data/lfw/tensorFeat.pkl','rb'))




class FewShotEpisodeGeneratorLabel(nn.Module):
    def __init__(self,n_labels_for_training,n_labels,n_samples_per_labels,n_train_labels,
                 n_val_labels,n_test_labels,n_q_per_label_train,n_q_per_label_val,n_q_per_label_test):
        super(FewShotEpisodeGeneratorLabel, self).__init__()
        
        self.n_labels_for_training=n_labels_for_training
        self.n_labels = n_labels
        self.n_samples_per_labels = n_samples_per_labels
        
        self.n_train_labels = n_train_labels
        self.n_val_labels= n_val_labels
        self.n_test_labels = n_test_labels
        
        self.n_q_per_label_train = n_q_per_label_train
        self.n_q_per_label_val = n_q_per_label_val
        self.n_q_per_label_test = n_q_per_label_test
        
    def getLabelList(self,startIndex,endIndexPlusOne,count):
        k = np.arange(startIndex,endIndexPlusOne)
        np.random.shuffle(k)
        return k[:count]
    
    def getTrainEpisode(self): 
        return self.getEpisode(1)

    def getValEpisode(self): 
        return self.getEpisode(2)

    def getTestEpisode(self): 
        return self.getEpisode(3)
    
    def getTestValEpisode(self): 
        return self.getEpisode(4)
    
    def getTrainonTestValEpisode(self):
        return self.getEpisode(5)
    
    def getTestValonTrainEpisode(self):
        return self.getEpisode(6)
    
    def getEpisode(self,mode):
        
        qr = []
        qrlabel = []
        qs = []
        qslabel = []
        
        # labels 0 to n_train_labels-1
        # choose n_labels from the set


        if(mode == 1):
            labelarr = self.getLabelList(0,self.n_train_labels,self.n_labels_for_training)
            q_per_label = self.n_q_per_label_train
        elif(mode == 2):
            labelarr = self.getLabelList(self.n_train_labels,
                                         self.n_train_labels+self.n_val_labels,self.n_labels)
            q_per_label = self.n_q_per_label_val
        elif(mode == 3):
            labelarr = self.getLabelList(self.n_train_labels+self.n_val_labels,
                                         self.n_train_labels+self.n_val_labels+self.n_test_labels,self.n_labels)
            q_per_label = self.n_q_per_label_test
            
        elif(mode == 4): # val and test combined
            labelarr = self.getLabelList(self.n_train_labels,
                                         self.n_train_labels+self.n_val_labels+self.n_test_labels,self.n_labels)
            q_per_label = self.n_q_per_label_test
            
        elif(mode == 5): # train on val and test combined
            labelarr = self.getLabelList(self.n_train_labels,
                                         self.n_train_labels+self.n_val_labels+self.n_test_labels,self.n_labels_for_training)
            q_per_label = self.n_q_per_label_train
            
        elif(mode == 6): # test on train
            labelarr = self.getLabelList(0,self.n_train_labels,self.n_labels)
            q_per_label = self.n_q_per_label_train

        
        for lb in labelarr:
            indPos = np.random.randint(0,len(attrListPos[lb]), 
                                       size=q_per_label+self.n_samples_per_labels)

            indSup = indPos[q_per_label:]
            indPos = indPos [:q_per_label]
            
            
            
            for hh in indPos:
                pos = attrListPos[lb][hh]
                tmp = facedatatrain[pos]
                qr.append(tmp.view(3,64,64))
                sampleLabel = labelcelebAzeroed[pos]
                freshlabel = 0
                if(mode==1 or mode ==5):
                    labelrow = torch.zeros(self.n_labels_for_training)
                else:
                    labelrow = torch.zeros(self.n_labels)
                for kk in labelarr:
                    if(sampleLabel[kk]==1):
                        labelrow[freshlabel] = 1
                    else:
                        labelrow[freshlabel] = 0
                    freshlabel = freshlabel + 1      
                qrlabel.append(labelrow)


            #Support
            for hh in indSup:
                pos = attrListPos[lb][hh]
                tmp = facedatatrain[pos]

                qs.append(tmp.view(3,64,64))
                sampleLabel = labelcelebAzeroed[pos]
                if(mode==1 or mode ==5):
                    labelrow = torch.zeros(self.n_labels_for_training)
                else:
                    labelrow = torch.zeros(self.n_labels)
                freshlabel = 0
                for kk in labelarr:
                    if(sampleLabel[kk]==1):
                        labelrow[freshlabel] = 1
                    else:
                        labelrow[freshlabel] = 0
                    freshlabel = freshlabel + 1
                        
                qslabel.append(labelrow)
        
        return torch.stack(qs),torch.stack(qslabel),torch.stack(qr),torch.stack(qrlabel),labelarr


# In[9]:

n_labels_for_training = args.train_way
n_labels= 5
n_samples_per_labels=5
n_train_labels=36
n_val_labels=18
n_test_labels=19
n_q_per_label_train = 5
n_q_per_label_val = 5
n_q_per_label_test = 5


# In[10]:


episodegen = FewShotEpisodeGeneratorLabel(n_labels_for_training,n_labels,n_samples_per_labels,n_train_labels,
                 n_val_labels,n_test_labels,n_q_per_label_train,n_q_per_label_val,n_q_per_label_test)




resnet50 = models.resnet50()
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(
            # stop at conv4
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4
        )
        self.fclast1 = nn.Linear(2048*4*4,1024)
        self.fclast2 = nn.Linear(1024,128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h1 = self.features(x)
        h1 = h1.view(-1,2048*4*4)
        h2 = self.relu(self.fclast1(h1))
        h3 = self.fclast2(h2)
        return h3

modelresnet50 = ResNet50()
if usecuda:
    modelresnet50.cuda(idgpu)


# Training will be done twice
# Phase 1 - train on first 36 labels and test on the remaining 37
# Phase 2 - train on last 37 labels and test on the first 36

print("Training will be done twice")
print("Phase 1 - train from scratch on first 36 labels and test on the remaining 37")
print("Phase 2 - train from scratch on last 37 labels and test on the first 36")


# Train on first 36 label and test on rest

print("Phase 1 starts----------Train on first 36 label and test on rest---------")

optimizer = optim.Adam(modelresnet50.parameters(), lr=1e-4)
avg_loss_train=[]
avg_loss_test=[]


def train(epoch):

    modelresnet50.train()
    train_loss = 0
    corCountTotal = 0 
    CountTotal = 0
    pdist = nn.PairwiseDistance(p=2)
    softmaxFunc = nn.Softmax()
    nllFunc = nn.NLLLoss()

    sumtotal = torch.zeros(1,total_labels)
    sumtotalcount = torch.zeros(1,total_labels)
    
    sumchosentruepositives = torch.zeros(1,total_labels)
    sumtruepositives = torch.zeros(1,total_labels)
    sumchosenpositives = torch.zeros(1,total_labels)

    for batch_idx in range(1000):
        supSet,supLabel,inp,inpLabel,reallabels = episodegen.getTrainEpisode()

        inp = Variable(inp)
        supSet = Variable(supSet)
        supLabel = Variable(supLabel)

        CountTotal = CountTotal + len(inp)

        if usecuda:
            inp = inp.cuda(idgpu)
            supSet = supSet.cuda(idgpu)
            supLabel = supLabel.cuda(idgpu)

        optimizer.zero_grad()

        inp_z=modelresnet50(inp)

        sup_z = modelresnet50(supSet)

        corCount = 0
        lossClass = 0

        tmp =0
        for inpi in range(len(inp)):
            actLabelList = inpLabel[inpi]

            z_one = inp_z[inpi]
            z_list = z_one.repeat(len(sup_z),1)

            distVect = pdist(z_list,sup_z) 
            distVect = -1*distVect
            distVect = distVect.view(-1,n_labels_for_training*n_samples_per_labels)

            for i in range(n_labels_for_training):
                fd = distVect[:,i*n_samples_per_labels:i*n_samples_per_labels+ n_samples_per_labels ]
                if(i ==0):
                    distVectAvg = torch.mean(torch.mean(fd))
                else:
                    distVectAvg = torch.cat((distVectAvg.view(-1),torch.mean(fd).view(-1)))

            distVectAvg = distVectAvg.view(-1,n_labels_for_training)

            target = actLabelList.view(1,n_labels_for_training)
            target = Variable(target)
            if usecuda:
                target = target.cuda(idgpu) 

            DTensor = distVect

            softmaxProb=softmaxFunc(DTensor)
            softmaxProb = softmaxProb.view(len(supSet),1)
            problabel = softmaxProb * supLabel
            problabel = torch.sum(problabel,0)

            problabelRounded = (problabel>=roundthreshold).type(torch.cuda.FloatTensor)

            if( torch.sum(problabelRounded == target).data.item() == n_labels_for_training):
                corCount = corCount + 1

            logloss = 0
            labelprod = problabel * target
            sumlogloss = 0
            sumcntlogloss = 0
            for poslabel in range(n_labels_for_training):
                if(labelprod[0,poslabel].data.item()>0):
                    sumlogloss = sumlogloss + -1*torch.log(labelprod[0,poslabel])
                    sumcntlogloss = sumcntlogloss +1

            if(sumcntlogloss>0):        
                logloss = sumlogloss/sumcntlogloss



            for lboutind in range(len(problabelRounded)):
                if(problabelRounded[lboutind].data.item() == actLabelList[lboutind]):#.cpu()
                    sumtotal[0,reallabels[lboutind]] = sumtotal[0,reallabels[lboutind]] + 1
                
                sumtotalcount[0,reallabels[lboutind]] = sumtotalcount[0,reallabels[lboutind]] + 1
                
                if(actLabelList[lboutind]==1):
                    sumtruepositives[0,reallabels[lboutind]] = sumtruepositives[0,reallabels[lboutind]]+1
                if(problabelRounded[lboutind].data.item()==1):
                    sumchosenpositives[0,reallabels[lboutind]] = sumchosenpositives[0,reallabels[lboutind]]+1
                if(problabelRounded[lboutind].data.item()==1 and actLabelList[lboutind]==1):
                    sumchosentruepositives[0,reallabels[lboutind]] = sumchosentruepositives[
                        0,reallabels[lboutind]]+1                



            if(inpi == 0):
                lossClass = logloss
            else:
                lossClass = lossClass + logloss


        lossClass = lossClass/len(inp)
        loss = lossClass 
        
        del supSet,supLabel,inp,inpLabel,reallabels
        if(type(loss)!=float):
        
            loss.backward()

            train_loss += loss.data.item()
            optimizer.step()
            batchloss = loss.data.item()
        else:
            batchloss = 0

        #Accuracy
        corCountTotal = corCountTotal + corCount      

        if batch_idx % loginterval == 0:
            print('Train Epoch: {} [{}/{} (Loss: {:.10f})'.format(
                epoch, batch_idx , 1000,batchloss ))


    train_loss = train_loss/1000

    print('Phase 1: Train============================> Epoch: {} Average Train loss: {:.20f}'.format(
        epoch, train_loss))

   


def testval(epoch):

    modelresnet50.eval()
    test_loss = 0
    corCountTotal = 0 
    CountTotal = 0
    pdist = nn.PairwiseDistance(p=2)
    softmaxFunc = nn.Softmax()
    nllFunc = nn.NLLLoss()    
    
    sumtotal = torch.zeros(1,total_labels)
    sumtotalcount = torch.zeros(1,total_labels)
    
    sumchosentruepositives = torch.zeros(1,total_labels)
    sumtruepositives = torch.zeros(1,total_labels)
    sumchosenpositives = torch.zeros(1,total_labels)
    
    for batch_idx in range(1000):
        
        supSet,supLabel,inp,inpLabel,reallabels = episodegen.getTestValEpisode()
        inp = Variable(inp)
        supSet = Variable(supSet)
        supLabel = Variable(supLabel)
        
        CountTotal = CountTotal + len(inp)
        
        if usecuda:
            inp = inp.cuda(idgpu)
            supSet = supSet.cuda(idgpu)
            supLabel = supLabel.cuda(idgpu)
            
            
        inp_z=modelresnet50(inp)
        
        sup_z = modelresnet50(supSet)

        corCount = 0
        lossClass = 0

        tmp =0
        
        for inpi in range(len(inp)):
            actLabelList = inpLabel[inpi]

            z_one = inp_z[inpi]
            z_list = z_one.repeat(len(sup_z),1)

            distVect = pdist(z_list,sup_z) 
            distVect = -1*distVect
            distVect = distVect.view(-1,n_labels*n_samples_per_labels)

            for i in range(n_labels):
                fd = distVect[:,i*n_samples_per_labels:i*n_samples_per_labels+ n_samples_per_labels ]
                if(i ==0):
                    distVectAvg = torch.mean(torch.mean(fd))
                else:
                    distVectAvg = torch.cat((distVectAvg.view(-1),torch.mean(fd).view(-1)))

            distVectAvg = distVectAvg.view(-1,n_labels)

            target = actLabelList.view(1,n_labels)
            target = Variable(target)
            if usecuda:
                target = target.cuda(idgpu)  

            DTensor = distVect

            softmaxProb=softmaxFunc(DTensor)
            softmaxProb = softmaxProb.view(len(supSet),1)
            problabel = softmaxProb * supLabel
            problabel = torch.sum(problabel,0)
            
            problabelRounded = (problabel>=roundthreshold).type(torch.cuda.FloatTensor)
            
        
            if( torch.sum(problabelRounded == target).data.item() == n_labels):
                corCount = corCount + 1
                
            logloss = 0
            labelprod = problabel * target
            sumlogloss = 0
            sumcntlogloss = 0
            for poslabel in range(n_labels):
                if(labelprod[0,poslabel].data.item()>0):
                    sumlogloss = sumlogloss + -1*torch.log(labelprod[0,poslabel])
                    sumcntlogloss = sumcntlogloss +1

            if(sumcntlogloss>0):        
                logloss = sumlogloss/sumcntlogloss
                
            
            for lboutind in range(len(problabelRounded)):
                if(problabelRounded[lboutind].data.item() == actLabelList[lboutind]): # .cpu()
                    sumtotal[0,reallabels[lboutind]] = sumtotal[0,reallabels[lboutind]] + 1
                    
                sumtotalcount[0,reallabels[lboutind]] = sumtotalcount[0,reallabels[lboutind]] + 1    
                    
                if(actLabelList[lboutind]==1):
                    sumtruepositives[0,reallabels[lboutind]] = sumtruepositives[0,reallabels[lboutind]]+1
                if(problabelRounded[lboutind].data.item()==1):
                    sumchosenpositives[0,reallabels[lboutind]] = sumchosenpositives[0,reallabels[lboutind]]+1
                if(problabelRounded[lboutind].data.item()==1 and actLabelList[lboutind]==1):
                    sumchosentruepositives[0,reallabels[lboutind]] = sumchosentruepositives[
                        0,reallabels[lboutind]]+1   


            if(inpi == 0):
                lossClass = logloss
            else:
                lossClass = lossClass + logloss

            
        lossClass = lossClass/len(inp)
        loss = lossClass
        supSet,supLabel,inp,inpLabel,reallabels
        if(type(loss)!=float):
            test_loss += loss.data.item()
            batchloss = loss.data.item()
        else:
            batchloss = loss
        
        corCountTotal = corCountTotal + corCount
        
        if batch_idx % loginterval == 0:
            print('Val Epoch: {} [{}/{} (Loss: {:.10f})'.format(
                epoch, batch_idx , 1000,batchloss))

    test_loss /= 1000

    print('Phase 1: Val============================> Epoch: {} Average Val loss: {:.20f}'.format(
        epoch, test_loss))

    F1_scores = 0
    print("------------------------------------------")

    # uncomment print statements to get the label wise F1 score
    for i in range(total_labels):
        if(i%10 ==0):
            #if(i>0):
            #    print()
            #print('F1 Score: ', end = ' ')
            dm =1
        if(sumchosenpositives[0,i] == 0):
            if(sumchosentruepositives[0,i]==0):
                #print('none,', end='')
                dm=1
            else:
                #print('error,',end='')
                dm=1
            continue
        else:
            prec = sumchosentruepositives[0,i]*100.0/sumchosenpositives[0,i]

        if(sumtruepositives[0,i] == 0):
            if(sumchosentruepositives[0,i]==0):
               #print('none,', end='')
               dm=1
            else:
               #print('error,',end='')
               dm=1
            continue
        else:
            recall = sumchosentruepositives[0,i]*100.0/sumtruepositives[0,i]

        if(recall+prec ==0):
            #print('divzeroerror,',end='')
            dm=1
        else:
            #print('{:.2f}%,'.format(2*(recall * prec) / (recall + prec)), end='')
            F1_scores = F1_scores + 2*(recall * prec) / (recall + prec)
    try:
        print("avg F1 scores: {:.4f}".format(F1_scores.item()/(100*n_train_labels)))
    except:
        print()
    print("------------------------------------------")
    print("------------------------------------------")
    print()
    return F1_scores
          
avg_F1_scores = 0
for epoch in range(1, epochs):
    t1 =  train(epoch)
    phase1_F1_scores = testval(epoch)
    

avg_F1_scores = avg_F1_scores + phase1_F1_scores


path =  './results/lfw/FSML{}shot/lfw_first_{}.torch'.format(args.shot,epoch)
torch.save(modelresnet50.state_dict(), path)


# Train on last 37 labels and test on the first 36 labels

print("Phase 2 starts----------Train on last 37 labels and test on rest---------")

# reinitialising the classifier
print("reinitialising the classifier")
modelresnet50 = ResNet50()
if usecuda:
    modelresnet50.cuda(idgpu)

optimizer = optim.Adam(modelresnet50.parameters(), lr=1e-4)
avg_loss_train=[]
avg_loss_test=[]


def trainrest(epoch):

    modelresnet50.train()
    train_loss = 0
    corCountTotal = 0 
    CountTotal = 0
    pdist = nn.PairwiseDistance(p=2)
    softmaxFunc = nn.Softmax()
    nllFunc = nn.NLLLoss()

    sumtotal = torch.zeros(1,total_labels)
    sumtotalcount = torch.zeros(1,total_labels)
    
    sumchosentruepositives = torch.zeros(1,total_labels)
    sumtruepositives = torch.zeros(1,total_labels)
    sumchosenpositives = torch.zeros(1,total_labels)

    for batch_idx in range(1000):
        supSet,supLabel,inp,inpLabel,reallabels = episodegen.getTrainonTestValEpisode()

        inp = Variable(inp)
        supSet = Variable(supSet)
        supLabel = Variable(supLabel)

        CountTotal = CountTotal + len(inp)

        if usecuda:
            inp = inp.cuda(idgpu)
            supSet = supSet.cuda(idgpu)
            supLabel = supLabel.cuda(idgpu)

        optimizer.zero_grad()

        inp_z=modelresnet50(inp)

        sup_z = modelresnet50(supSet)

        corCount = 0
        lossClass = 0

        tmp =0
        for inpi in range(len(inp)):
            actLabelList = inpLabel[inpi]

            z_one = inp_z[inpi]
            z_list = z_one.repeat(len(sup_z),1)

            distVect = pdist(z_list,sup_z) 
            distVect = -1*distVect
            distVect = distVect.view(-1,n_labels_for_training*n_samples_per_labels)


            for i in range(n_labels_for_training):
                fd = distVect[:,i*n_samples_per_labels:i*n_samples_per_labels+ n_samples_per_labels ]
                if(i ==0):
                    distVectAvg = torch.mean(torch.mean(fd))
                else:
                    distVectAvg = torch.cat((distVectAvg.view(-1),torch.mean(fd).view(-1)))

            distVectAvg = distVectAvg.view(-1,n_labels_for_training)

            target = actLabelList.view(1,n_labels_for_training)
            target = Variable(target)
            if usecuda:
                target = target.cuda(idgpu)
 

            DTensor = distVect

            softmaxProb=softmaxFunc(DTensor)
            softmaxProb = softmaxProb.view(len(supSet),1)
            problabel = softmaxProb * supLabel
            problabel = torch.sum(problabel,0)

            problabelRounded = (problabel>=roundthreshold).type(torch.cuda.FloatTensor)


            if( torch.sum(problabelRounded == target).data.item() == n_labels_for_training):
                corCount = corCount + 1

            logloss = 0
            labelprod = problabel * target
            sumlogloss = 0
            sumcntlogloss = 0
            for poslabel in range(n_labels_for_training):
                if(labelprod[0,poslabel].data.item()>0):
                    sumlogloss = sumlogloss + -1*torch.log(labelprod[0,poslabel])
                    sumcntlogloss = sumcntlogloss +1

            if(sumcntlogloss>0):        
                logloss = sumlogloss/sumcntlogloss



            for lboutind in range(len(problabelRounded)):
                if(problabelRounded[lboutind].data.item() == actLabelList[lboutind]):#.cpu()
                    sumtotal[0,reallabels[lboutind]] = sumtotal[0,reallabels[lboutind]] + 1
                
                sumtotalcount[0,reallabels[lboutind]] = sumtotalcount[0,reallabels[lboutind]] + 1
                
                if(actLabelList[lboutind]==1):
                    sumtruepositives[0,reallabels[lboutind]] = sumtruepositives[0,reallabels[lboutind]]+1
                if(problabelRounded[lboutind].data.item()==1):
                    sumchosenpositives[0,reallabels[lboutind]] = sumchosenpositives[0,reallabels[lboutind]]+1
                if(problabelRounded[lboutind].data.item()==1 and actLabelList[lboutind]==1):
                    sumchosentruepositives[0,reallabels[lboutind]] = sumchosentruepositives[
                        0,reallabels[lboutind]]+1                



            if(inpi == 0):
                lossClass = logloss
            else:
                lossClass = lossClass + logloss

        lossClass = lossClass/len(inp)
        loss = lossClass
        
        del supSet,supLabel,inp,inpLabel,reallabels
        if(type(loss)!=float):
        
            loss.backward()

            train_loss += loss.data.item()
            optimizer.step()
            batchloss = loss.data.item()
        else:
            batchloss = 0

        #Accuracy
        corCountTotal = corCountTotal + corCount      

        if batch_idx % loginterval == 0:
            print('Train Epoch: {} [{}/{} (Loss: {:.10f})'.format(
                epoch, batch_idx , 1000,batchloss ))
           

    train_loss = train_loss/1000

    print('Phase 2: Train============================> Epoch: {} Average Train loss: {:.20f}'.format(
        epoch, train_loss))

  


def testvalrest(epoch):
    modelresnet50.eval()
    test_loss = 0
    corCountTotal = 0 
    CountTotal = 0
    pdist = nn.PairwiseDistance(p=2)
    softmaxFunc = nn.Softmax()
    nllFunc = nn.NLLLoss()    
    
    sumtotal = torch.zeros(1,total_labels)
    sumtotalcount = torch.zeros(1,total_labels)
    
    sumchosentruepositives = torch.zeros(1,total_labels)
    sumtruepositives = torch.zeros(1,total_labels)
    sumchosenpositives = torch.zeros(1,total_labels)
    
    for batch_idx in range(1000):
        
        supSet,supLabel,inp,inpLabel,reallabels = episodegen.getTestValonTrainEpisode()
        inp = Variable(inp)
        supSet = Variable(supSet)
        supLabel = Variable(supLabel)
        
        CountTotal = CountTotal + len(inp)
        
        if usecuda:
            inp = inp.cuda(idgpu)
            supSet = supSet.cuda(idgpu)
            supLabel = supLabel.cuda(idgpu)
            
            
        inp_z=modelresnet50(inp)
        
        sup_z = modelresnet50(supSet)

        corCount = 0
        lossClass = 0

        tmp =0
        
        for inpi in range(len(inp)):
            actLabelList = inpLabel[inpi]

            z_one = inp_z[inpi]
            z_list = z_one.repeat(len(sup_z),1)

            distVect = pdist(z_list,sup_z) 
            distVect = -1*distVect
            distVect = distVect.view(-1,n_labels*n_samples_per_labels)


            for i in range(n_labels):
                fd = distVect[:,i*n_samples_per_labels:i*n_samples_per_labels+ n_samples_per_labels ]
                if(i ==0):
                    distVectAvg = torch.mean(torch.mean(fd))
                else:
                    distVectAvg = torch.cat((distVectAvg.view(-1),torch.mean(fd).view(-1)))

            distVectAvg = distVectAvg.view(-1,n_labels)

            target = actLabelList.view(1,n_labels)
            target = Variable(target)
            if usecuda:
                target = target.cuda(idgpu)


            DTensor = distVect

            softmaxProb=softmaxFunc(DTensor)
            softmaxProb = softmaxProb.view(len(supSet),1)
            problabel = softmaxProb * supLabel
            problabel = torch.sum(problabel,0)
            
            problabelRounded = (problabel>=roundthreshold).type(torch.cuda.FloatTensor)
        
            if( torch.sum(problabelRounded == target).data.item() == n_labels):
                corCount = corCount + 1
                
            logloss = 0
            labelprod = problabel * target
            sumlogloss = 0
            sumcntlogloss = 0
            for poslabel in range(n_labels):
                if(labelprod[0,poslabel].data.item()>0):
                    sumlogloss = sumlogloss + -1*torch.log(labelprod[0,poslabel])
                    sumcntlogloss = sumcntlogloss +1

            if(sumcntlogloss>0):        
                logloss = sumlogloss/sumcntlogloss
                
            
            for lboutind in range(len(problabelRounded)):
                if(problabelRounded[lboutind].data.item()== actLabelList[lboutind]): #.cpu() 
                    sumtotal[0,reallabels[lboutind]] = sumtotal[0,reallabels[lboutind]] + 1
                    
                sumtotalcount[0,reallabels[lboutind]] = sumtotalcount[0,reallabels[lboutind]] + 1    
                    
                if(actLabelList[lboutind]==1):
                    sumtruepositives[0,reallabels[lboutind]] = sumtruepositives[0,reallabels[lboutind]]+1
                if(problabelRounded[lboutind].data.item()==1):
                    sumchosenpositives[0,reallabels[lboutind]] = sumchosenpositives[0,reallabels[lboutind]]+1
                if(problabelRounded[lboutind].data.item()==1 and actLabelList[lboutind]==1):
                    sumchosentruepositives[0,reallabels[lboutind]] = sumchosentruepositives[
                        0,reallabels[lboutind]]+1   


            if(inpi == 0):
                lossClass = logloss
            else:
                lossClass = lossClass + logloss

            
        lossClass = lossClass/len(inp)
        loss = lossClass
        supSet,supLabel,inp,inpLabel,reallabels
        if(type(loss)!=float):
            test_loss += loss.data.item()
            batchloss = loss.data.item()
        else:
            batchloss = loss
        
        corCountTotal = corCountTotal + corCount
        
        if batch_idx % loginterval == 0:
            print('Val Epoch: {} [{}/{} (Loss: {:.10f})'.format(
                epoch, batch_idx , 1000,batchloss))
           
    test_loss /= 1000

    print('Phase 2: Val============================> Epoch: {} Average Val loss: {:.20f}'.format(
        epoch, test_loss))

    print("------------------------------------------")

    # uncomment print statements to get the label wise F1 score

    F1_scores = 0
    for i in range(total_labels):
        if(i%10 ==0):
            #if(i>0):
            #    print()
            #print('F1 Score: ', end = ' ')
            dm=1
        if(sumchosenpositives[0,i] == 0):
            if(sumchosentruepositives[0,i]==0):
                #print('none,', end='')
                dm=1
            else:
                #print('error,',end='')
                dm=1
            continue
        else:
            prec = sumchosentruepositives[0,i]*100.0/sumchosenpositives[0,i]

        if(sumtruepositives[0,i] == 0):
            if(sumchosentruepositives[0,i]==0):
               #print('none,', end='')
               dm=1
            else:
               #print('error,',end='')
               dm=1
            continue
        else:
            recall = sumchosentruepositives[0,i]*100.0/sumtruepositives[0,i]

        if(recall+prec ==0):
            #print('divzeroerror,',end='')
            dm=1
        else:
            #print('{:.2f}%,'.format(2*(recall * prec) / (recall + prec)), end='')
            F1_scores = F1_scores + 2*(recall * prec) / (recall + prec)
            
    try:
        print("avg F1 scores: {:.4f}".format(F1_scores.item()/(100*(n_val_labels+n_test_labels))))
    except:
        print()
    print("------------------------------------------")
    print("------------------------------------------")
    print()
    return F1_scores
    
          




for epoch in range(1, epochs):
    trainrest(epoch)
    phase2_F1_scores = testvalrest(epoch)

avg_F1_scores = avg_F1_scores + phase2_F1_scores

avg_F1_scores = avg_F1_scores/total_labels
print()
print('Average Test F1 Score after {} epochs = {:.4f}'.format(epochs-1,avg_F1_scores))

path =  './results/lfw/FSML{}shot/lfw_last_{}.torch'.format(args.shot,epoch)
torch.save(modelresnet50.state_dict(), path)


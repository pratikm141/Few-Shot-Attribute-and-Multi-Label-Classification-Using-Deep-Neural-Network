from __future__ import print_function
from collections import defaultdict
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms

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
args = parser.parse_args()

if(args.epochs<2):
    print("--epochs should be more than 1")
    os._exit(1)

print('gpu ',args.gpu,' shot=',args.shot)

idgpu = args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(idgpu)
idgpu = 0
usecuda = not args.no_cuda and torch.cuda.is_available()
batch_size = 64
epochs = args.epochs
seed = 1
loginterval= 200
total_labels= 40
torch.manual_seed(seed)
if usecuda:
    torch.cuda.manual_seed(seed)
    
kwargs = {}

directory = './results/celeba/base{}shot'.format(args.shot)
if not os.path.exists(directory):
    os.makedirs(directory)


import pickle
labelcelebA = pickle.load(open('./data/celeba/labelcelebA.pkl','rb'))
labelcelebAzeroed = (labelcelebA + 1)/2


import tqdm
#only positive indices rangettr kept
attrListPos = []
attrListNeg = []
for i in tqdm.tqdm(range(40)):
    attrPos = []
    attrNeg = []
    for j in range(len(labelcelebA[:,i])):
        if(labelcelebA[j,i]>=0):
            attrPos.append(j)
        else:
            attrNeg.append(j)

    attrListPos.append(attrPos)
    attrListNeg.append(attrNeg)



facedatatrain =datasets.ImageFolder('./data/celeba/celeba_img',
                                    transforms.Compose([transforms.ToTensor()]))



class FewShotEpisodeGeneratorLabel(nn.Module):
    def __init__(self,n_labels,n_samples_per_labels,n_train_labels,
                 n_val_labels,n_test_labels,n_q_per_label_train,n_q_per_label_val,n_q_per_label_test):
        super(FewShotEpisodeGeneratorLabel, self).__init__()
        
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
            labelarr = self.getLabelList(0,self.n_train_labels,self.n_labels)
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
                                         self.n_train_labels+self.n_val_labels+self.n_test_labels,self.n_labels)
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
                tmp = tmp[0]
                qr.append(tmp.view(3,64,64))
                sampleLabel = labelcelebAzeroed[pos]
                freshlabel = 0
                labelrow = torch.zeros(self.n_labels)
                for kk in labelarr:
                    if(sampleLabel[kk]==1 and lb == kk):
                        labelrow[freshlabel] = 1
                    else:
                        labelrow[freshlabel] = 0
                    freshlabel = freshlabel + 1       
                qrlabel.append(labelrow)


            #Support
            for hh in indSup:
                pos = attrListPos[lb][hh]
                tmp = facedatatrain[pos]
                tmp = tmp[0]
                qs.append(tmp.view(3,64,64))
                sampleLabel = labelcelebAzeroed[pos]
                labelrow = torch.zeros(self.n_labels)
                freshlabel = 0
                for kk in labelarr:
                    if(sampleLabel[kk]==1 and lb == kk):
                        labelrow[freshlabel] = 1
                    else:
                        labelrow[freshlabel] = 0
                    freshlabel = freshlabel + 1
                        
                qslabel.append(labelrow)
        
        return torch.stack(qs),torch.stack(qslabel),torch.stack(qr),torch.stack(qrlabel),labelarr


n_labels= 5
n_samples_per_labels=args.shot
n_train_labels=20
n_val_labels=10
n_test_labels=10
n_q_per_label_train = 5
n_q_per_label_val = 5
n_q_per_label_test = 5



episodegen = FewShotEpisodeGeneratorLabel(n_labels,n_samples_per_labels,n_train_labels,
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
# Phase 1 - train on first 20 labels and test on the remaining 20
# Phase 2 - train on last 20 labels and test on the first 20

print("Training will be done twice")
print("Phase 1 - train from scratch on first 20 labels and test on the remaining 20")
print("Phase 2 - train from scratch on last 20 labels and test on the first 20")


# Train on first 20 label and test on rest

print("Phase 1 starts----------Train on first 20 label and test on rest---------")

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

    sumtotal = torch.zeros(1,40)
    sumtotalcount = torch.zeros(1,40)

    sumchosentruepositives = torch.zeros(1,40)
    sumtruepositives = torch.zeros(1,40)
    sumchosenpositives = torch.zeros(1,40)
    
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

        z_proto_list = []
        #prototype z
        for pt in range(n_labels):
            z_proto_list.append(torch.mean(
                sup_z[pt*n_samples_per_labels:pt*n_samples_per_labels+n_samples_per_labels],0))

        z_proto = torch.stack(z_proto_list)

        corCount = 0
        lossClass = 0

        tmp =0
        
        for inpi in range(len(inp)):
            actLabelList = inpLabel[inpi]

            z_one = inp_z[inpi]
            z_list = z_one.repeat(len(z_proto),1)

            distVect = pdist(z_list,z_proto) 

            distVect = distVect.view(-1,5)

            softmaxProb = F.softmax(-distVect).view(n_labels, 1)

            val,pos =torch.max(softmaxProb,0)

            problabelRounded = torch.zeros(1,n_labels)
            problabelRounded[0,pos.data.item()] = 1

            # problabelRounded = (problabel>=0.5).type(torch.cuda.FloatTensor)

            target = actLabelList.view(n_labels,1)
            target = Variable(target)
            if usecuda:
                target = target.cuda(idgpu)

            labelprod = softmaxProb*target

            cellsum = 0
            for lp in range(n_labels):
                if(labelprod[lp,0].data.item()>0):
                    cellsum = cellsum + (-1*torch.log(labelprod[lp,0]))
            logloss =  cellsum/n_labels

            if( torch.sum(problabelRounded == actLabelList) == n_labels):
                corCount = corCount + 1

            for lboutind in range(len(problabelRounded)):
                if(problabelRounded[0,lboutind] == actLabelList[lboutind]):
                    sumtotal[0,reallabels[lboutind]] = sumtotal[0,reallabels[lboutind]] + 1

                sumtotalcount[0,reallabels[lboutind]] = sumtotalcount[0,reallabels[lboutind]] + 1

                if(actLabelList[lboutind]==1):
                    sumtruepositives[0,reallabels[lboutind]] = sumtruepositives[0,reallabels[lboutind]]+1
                if(problabelRounded[0,lboutind]==1):
                    sumchosenpositives[0,reallabels[lboutind]] = sumchosenpositives[0,reallabels[lboutind]]+1
                if(problabelRounded[0,lboutind]==1 and actLabelList[lboutind]==1):
                    sumchosentruepositives[0,reallabels[lboutind]] = sumchosentruepositives[
                        0,reallabels[lboutind]]+1              



            if(inpi == 0):
                lossClass = logloss
            else:
                lossClass = lossClass + logloss
                
        lossClass = lossClass/len(inp)
        loss = lossClass
        
        
        if(type(loss)!=float):
            loss.backward()
            batchloss = loss.data.item()
            train_loss += loss.data.item()
            optimizer.step()
        else:
            batchloss = loss
            

        #Accuracy
        corCountTotal = corCountTotal + corCount      

        if batch_idx % loginterval == 0:
            print('Train Epoch: {} [{}/{} (Loss: {:.20f})'.format(
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
    
    sumtotal = torch.zeros(1,40)
    sumtotalcount = torch.zeros(1,40)
    
    sumchosentruepositives = torch.zeros(1,40)
    sumtruepositives = torch.zeros(1,40)
    sumchosenpositives = torch.zeros(1,40)
    
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
        
        z_proto_list = []
        #prototype z
        for pt in range(n_labels):
            z_proto_list.append(torch.mean(
                sup_z[pt*n_samples_per_labels:pt*n_samples_per_labels+n_samples_per_labels],0))

        z_proto = torch.stack(z_proto_list)

        corCount = 0
        lossClass = 0

        tmp =0
        
        for inpi in range(len(inp)):
            actLabelList = inpLabel[inpi]

            z_one = inp_z[inpi]
            z_list = z_one.repeat(len(z_proto),1)

            distVect = pdist(z_list,z_proto) 

            distVect = distVect.view(-1,5)

            softmaxProb = F.softmax(-distVect).view(n_labels, 1)

            val,pos =torch.max(softmaxProb,0)

            problabelRounded = torch.zeros(1,n_labels)
            problabelRounded[0,pos.data.item()] = 1

            target = actLabelList.view(n_labels,1)
            target = Variable(target)
            if usecuda:
                target = target.cuda(idgpu)

            labelprod = softmaxProb*target

            cellsum = 0
            for lp in range(n_labels):
                if(labelprod[lp,0].data.item()>0):
                    cellsum = cellsum + labelprod[lp,0]

            logloss =  cellsum/n_labels


            if( torch.sum(problabelRounded == actLabelList) == n_labels):
                corCount = corCount + 1

            for lboutind in range(len(problabelRounded)):
                if(problabelRounded[0,lboutind] == actLabelList[lboutind]):
                    sumtotal[0,reallabels[lboutind]] = sumtotal[0,reallabels[lboutind]] + 1

                sumtotalcount[0,reallabels[lboutind]] = sumtotalcount[0,reallabels[lboutind]] + 1

                if(actLabelList[lboutind]==1):
                    sumtruepositives[0,reallabels[lboutind]] = sumtruepositives[0,reallabels[lboutind]]+1
                if(problabelRounded[0,lboutind]==1):
                    sumchosenpositives[0,reallabels[lboutind]] = sumchosenpositives[0,reallabels[lboutind]]+1
                if(problabelRounded[0,lboutind]==1 and actLabelList[lboutind]==1):
                    sumchosentruepositives[0,reallabels[lboutind]] = sumchosentruepositives[
                        0,reallabels[lboutind]]+1                



            if(inpi == 0):
                lossClass = logloss
            else:
                lossClass = lossClass + logloss
            
        lossClass = lossClass/len(inp)
        loss = lossClass
        
        if(type(loss)!=float):
            test_loss += loss.data.item()
            batchloss = loss.data.item()
        else:
            batchloss = 0
        
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

avg_F1_scores = 0
for epoch in range(1, epochs):
    t1 =  train(epoch)
    phase1_F1_scores = testval(epoch)
    

avg_F1_scores = avg_F1_scores + phase1_F1_scores

path =  './results/celeba/base{}shot/celeba_first_{}.torch'.format(args.shot,epoch)
torch.save(modelresnet50.state_dict(), path)


# Train on last 20 labels and test on the first 20 labels

print("Phase 2 starts----------Train on last 20 labels and test on rest---------")

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

    sumtotal = torch.zeros(1,40)
    sumtotalcount = torch.zeros(1,40)

    sumchosentruepositives = torch.zeros(1,40)
    sumtruepositives = torch.zeros(1,40)
    sumchosenpositives = torch.zeros(1,40)
    
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

        z_proto_list = []
        #prototype z
        for pt in range(n_labels):
            z_proto_list.append(torch.mean(
                sup_z[pt*n_samples_per_labels:pt*n_samples_per_labels+n_samples_per_labels],0))

        z_proto = torch.stack(z_proto_list)

        corCount = 0
        lossClass = 0

        tmp =0
        
        for inpi in range(len(inp)):
            actLabelList = inpLabel[inpi]

            z_one = inp_z[inpi]
            z_list = z_one.repeat(len(z_proto),1)

            distVect = pdist(z_list,z_proto) 

            distVect = distVect.view(-1,5)

            softmaxProb = F.softmax(-distVect).view(n_labels, 1)

            val,pos =torch.max(softmaxProb,0)

            problabelRounded = torch.zeros(1,n_labels)
            problabelRounded[0,pos.data.item()] = 1

            target = actLabelList.view(n_labels,1)
            target = Variable(target)
            if usecuda:
                target = target.cuda(idgpu)

            labelprod = softmaxProb*target

            cellsum = 0
            for lp in range(n_labels):
                if(labelprod[lp,0].data.item()>0):
                    cellsum = cellsum + (-1*torch.log(labelprod[lp,0]))
            logloss =  cellsum/n_labels

            if( torch.sum(problabelRounded == actLabelList) == n_labels):
                corCount = corCount + 1
            

            for lboutind in range(len(problabelRounded)):
                if(problabelRounded[0,lboutind] == actLabelList[lboutind]):
                    sumtotal[0,reallabels[lboutind]] = sumtotal[0,reallabels[lboutind]] + 1

                sumtotalcount[0,reallabels[lboutind]] = sumtotalcount[0,reallabels[lboutind]] + 1

                if(actLabelList[lboutind]==1):
                    sumtruepositives[0,reallabels[lboutind]] = sumtruepositives[0,reallabels[lboutind]]+1
                if(problabelRounded[0,lboutind]==1):
                    sumchosenpositives[0,reallabels[lboutind]] = sumchosenpositives[0,reallabels[lboutind]]+1
                if(problabelRounded[0,lboutind]==1 and actLabelList[lboutind]==1):
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
            batchloss = loss.data.item()
            train_loss += loss.data.item()
            optimizer.step()
        else:
            batchloss = loss

        #Accuracy
        corCountTotal = corCountTotal + corCount      

        if batch_idx % loginterval == 0:
            print('Train Epoch: {} [{}/{} (Loss: {:.20f})'.format(
                epoch, batch_idx , 1000,batchloss ))

    train_loss = train_loss/1000

    print('Phase 2: Train============================> Epoch: {} Average Train loss: {:.20f}'.format(
        epoch, train_loss,corCountTotal*100.0/CountTotal))

 

def testvalrest(epoch):
    modelresnet50.eval()
    test_loss = 0
    corCountTotal = 0 
    CountTotal = 0
    pdist = nn.PairwiseDistance(p=2)
    softmaxFunc = nn.Softmax()
    nllFunc = nn.NLLLoss()    
    
    sumtotal = torch.zeros(1,40)
    sumtotalcount = torch.zeros(1,40)
    
    sumchosentruepositives = torch.zeros(1,40)
    sumtruepositives = torch.zeros(1,40)
    sumchosenpositives = torch.zeros(1,40)
    
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
        
        z_proto_list = []
        #prototype z
        for pt in range(n_labels):
            z_proto_list.append(torch.mean(
                sup_z[pt*n_samples_per_labels:pt*n_samples_per_labels+n_samples_per_labels],0))

        z_proto = torch.stack(z_proto_list)

        corCount = 0
        lossClass = 0

        tmp =0
        
        for inpi in range(len(inp)):
            actLabelList = inpLabel[inpi]

            z_one = inp_z[inpi]
            z_list = z_one.repeat(len(z_proto),1)

            distVect = pdist(z_list,z_proto) 

            distVect = distVect.view(-1,5)

            softmaxProb = F.softmax(-distVect).view(n_labels, 1)

            val,pos =torch.max(softmaxProb,0)

            problabelRounded = torch.zeros(1,n_labels)
            problabelRounded[0,pos.data.item()] = 1

            target = actLabelList.view(n_labels,1)
            target = Variable(target)
            if usecuda:
                target = target.cuda(idgpu)

            labelprod = softmaxProb*target

            cellsum = 0
            for lp in range(n_labels):
                if(labelprod[lp,0].data.item()>0):
                    cellsum = cellsum + labelprod[lp,0]

            logloss =  cellsum/n_labels

            if( torch.sum(problabelRounded == actLabelList) == n_labels):
                corCount = corCount + 1

            for lboutind in range(len(problabelRounded)):
                if(problabelRounded[0,lboutind] == actLabelList[lboutind]):
                    sumtotal[0,reallabels[lboutind]] = sumtotal[0,reallabels[lboutind]] + 1

                sumtotalcount[0,reallabels[lboutind]] = sumtotalcount[0,reallabels[lboutind]] + 1

                if(actLabelList[lboutind]==1):
                    sumtruepositives[0,reallabels[lboutind]] = sumtruepositives[0,reallabels[lboutind]]+1
                if(problabelRounded[0,lboutind]==1):
                    sumchosenpositives[0,reallabels[lboutind]] = sumchosenpositives[0,reallabels[lboutind]]+1
                if(problabelRounded[0,lboutind]==1 and actLabelList[lboutind]==1):
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
            test_loss += loss.data.item()
            batchloss = loss.data.item()
        else:
            batchloss = 0
        
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
        print("avg F1 scores: {:.4f}".format(F1_scores.item()/(100*n_train_labels)))
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

avg_F1_scores = avg_F1_scores/(100*total_labels)
print()
print('Average Test F1 Score after {} epochs = {:.4f}'.format(epochs-1,avg_F1_scores))

path =  './results/celeba/base{}shot/celeba_last_{}.torch'.format(args.shot,epoch)
torch.save(modelresnet50.state_dict(), path)



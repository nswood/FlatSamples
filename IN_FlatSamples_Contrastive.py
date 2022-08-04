# Imports basics

import numpy as np
import h5py
import json
import setGPU

# Imports neural net tools
import itertools
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
from fast_soft_sort.pytorch_ops import soft_rank


# Opens files and reads data

print("Extracting")
outdir = 'data/IN_FlatSamples_Pytorch'
label='contrastive'
fOne = h5py.File("data/FullQCD_FullSig_Zqq_fillfactor1_pTsdmassfilling_dRlimit08_50particlesordered_sigFill_genMatched50.h5", 'r')
totalData = fOne["deepDoubleQ"][:]
print(totalData.shape)

# Sets controllable values

particlesConsidered = 50
particlesPostCut = 50
entriesPerParticle = 4
eventDataLength = 6
decayTypeColumn = -1
datapoints = 10000
trainingDataLength = int(len(totalData)*0.8)
validationDataLength = int(len(totalData)*0.1)
batchSize = 128

includeeventData = False

modelName = "IN_FlatSamples_EighthQCDEighthSig_50particles_pTsdmassfilling_dRlimit08"

# Creates Training Data

print("Preparing Data")

particleDataLength = particlesConsidered * entriesPerParticle

np.random.seed(42)
np.random.shuffle(totalData)

print(totalData.shape)
trainingDataLength = int(len(totalData)*0.8)
validationDataLength = int(len(totalData)*0.1)


labels = totalData[:, decayTypeColumn:]
particleData = totalData[:, eventDataLength:particleDataLength + eventDataLength]
eventData = totalData[:, :eventDataLength]
jetMassData = totalData[:, eventDataLength-1] #last entry in eventData (zero indexing)

# Training Data
eventTrainingData = np.array(eventData[0:trainingDataLength])
jetMassTrainingData = np.array(jetMassData[0:trainingDataLength])
particleTrainingData = np.transpose(
    particleData[0:trainingDataLength, ].reshape(trainingDataLength, 
                                                 entriesPerParticle, 
                                                 particlesConsidered),
                                                 axes=(0, 1, 2))
trainingLabels = np.array([[i, 1-i] for i in labels[0:trainingDataLength]]).reshape((-1, 2))

# Validation Data
eventValidationData = np.array(eventData[trainingDataLength:trainingDataLength + validationDataLength])
jetMassValidationData = np.array(jetMassData[trainingDataLength:trainingDataLength + validationDataLength])
particleValidationData = np.transpose(
    particleData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength,
                                                                                         entriesPerParticle,
                                                                                         particlesConsidered),
                                                                                         axes=(0, 1, 2))
validationLabels = np.array([[i, 1-i] for i in labels[trainingDataLength:trainingDataLength + validationDataLength]]).reshape((-1, 2))

# Testing Data
particleTestData = np.transpose(particleData[trainingDataLength + validationDataLength:,].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerParticle, particlesConsidered),
                                axes=(0, 1, 2))
testLabels = np.array(labels[trainingDataLength + validationDataLength:])

print('Selecting particlesPostCut')
particleTrainingData = particleTrainingData[:, :particlesPostCut]
particleValidationData = particleValidationData[:, :particlesPostCut]


# Separating signal and bkg arrays
particleTrainingDataSig = particleTrainingData[[trainingLabels[:,0].astype(bool)]]
particleTrainingDataBkg = particleTrainingData[trainingLabels[:,1].astype(bool)]
particleValidationDataSig = particleValidationData[validationLabels[:,0].astype(bool)]
particleValidationDataBkg = particleValidationData[validationLabels[:,1].astype(bool)]


# Jet mass for correlation
jetMassTrainingDataSig = jetMassTrainingData[trainingLabels[:,0].astype(bool)]
jetMassTrainingDataBkg = jetMassTrainingData[trainingLabels[:,1].astype(bool)]
jetMassValidationDataSig = jetMassValidationData[validationLabels[:,0].astype(bool)]
jetMassValidationDataBkg = jetMassValidationData[validationLabels[:,1].astype(bool)]

# Defines the interaction matrices
particlesConsidered = particlesPostCut

class GraphNetnoSV(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden, De=5, Do=6, softmax=False):
        super(GraphNetnoSV, self).__init__()
        self.hidden = int(hidden)
        self.P = params
        self.Nv = 0 
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.Ns = self.Nv * (self.Nv - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.S = 0
        self.n_targets = n_targets
        self.assign_matrices()
           
        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).cuda()
        self.fr2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3 = nn.Linear(int(self.hidden/2), self.De).cuda()
        self.fr1_pv = nn.Linear(self.S + self.P + self.Dr, self.hidden).cuda()
        self.fr2_pv = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3_pv = nn.Linear(int(self.hidden/2), self.De).cuda()
        
        self.fo1 = nn.Linear(self.P + self.Dx + (self.De), self.hidden).cuda()
        self.fo2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fo3 = nn.Linear(int(self.hidden/2), self.Do).cuda()
        
        self.fc_fixed = nn.Linear(self.Do, self.n_targets).cuda()
            
    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = (self.Rr).cuda()
        self.Rs = (self.Rs).cuda()

    def forward(self, x):
        ###PF Candidate - PF Candidate###
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
        B = nn.functional.relu(self.fr2(B))
        E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pp = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        

        ####Final output matrix for particles###
        

        C = torch.cat([x, Ebar_pp], 1)
        del Ebar_pp
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + (self.De))))
        C = nn.functional.relu(self.fo2(C))
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C

        
        #Taking the sum of over each particle/vertex
        N = torch.sum(O, dim=1)
        del O
        
        ### Classification MLP ###

        N = self.fc_fixed(N)
        
        if softmax:
            N = nn.Softmax(dim=1)(N)

        return N 
            
    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])

class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = torch.device('cuda:0')

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        #self.device = (torch.device('cuda')if z_a.is_cuda else torch.device('cpu'))
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()
        return loss

class CorrLoss(nn.Module):
    def __init__(self, corr=False,sort_tolerance=1.0,sort_reg='l2'):
        super(CorrLoss, self).__init__()
        self.tolerance = sort_tolerance
        self.reg       = sort_reg
        self.corr      = corr
        
    def spearman(self, pred, target):
        pred   = soft_rank(pred.cpu().reshape(1,-1),regularization=self.reg,regularization_strength=self.tolerance,)
        target = soft_rank(target.cpu().reshape(1,-1),regularization=self.reg,regularization_strength=self.tolerance,)
        #pred   = torchsort.soft_rank(pred.reshape(1,-1),regularization_strength=x)
        #target = torchsort.soft_rank(target.reshape(1,-1),regularization_strength=x)
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()
        ret = (pred * target).sum()
        if self.corr:
            return (1-ret)*(1-ret)
        else:
            return ret*ret
    
    def forward(self, features, labels):
        return self.spearman(features,labels)
    
##############################################   
########## DUC I HAVE TO PEE #################
##############################################


n_targets = 2
gnn = GraphNetnoSV(particlesPostCut, n_targets, entriesPerParticle, 15,
                       De=5,
                       Do=6, softmax=False)
        
n_epochs = 200
    
loss = nn.BCELoss(reduction='mean')
clr_criterion  = BarlowTwinsLoss(lambda_param=1.0)
cor_criterion  = CorrLoss()
acr_criterion  = CorrLoss(corr=True)

BarlowLoss = True

optimizer = optim.Adam(gnn.parameters(), lr = 0.0001)
loss_vals_training = np.zeros(n_epochs)
loss_std_training = np.zeros(n_epochs)
loss_vals_validation = np.zeros(n_epochs)
loss_std_validation = np.zeros(n_epochs)

acc_vals_training = np.zeros(n_epochs)
acc_vals_validation = np.zeros(n_epochs)
acc_std_training = np.zeros(n_epochs)
acc_std_validation = np.zeros(n_epochs)

final_epoch = 0
l_val_best = 99999

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
softmax = torch.nn.Softmax(dim=1)
import time
from tqdm import tqdm 

for m in range(n_epochs):
    print("Epoch %s\n" % m)
    #torch.cuda.empty_cache()
    final_epoch = m
    lst = []
    loss_val = []
    loss_training = []
    correct = []
    tic = time.perf_counter()
   
    for i in tqdm(range(int(0.8*datapoints/batchSize))): 
        #print('%s out of %s'%(i, int(particleTrainingData.shape[0]/batchSize)))
        
        optimizer.zero_grad()
        trainingvSig = torch.FloatTensor(particleTrainingDataSig[i*batchSize:(i+1)*batchSize]).cuda()
        trainingvBkg = torch.FloatTensor(particleTrainingDataBkg[i*batchSize:(i+1)*batchSize]).cuda()
        trainingvMass = torch.FloatTensor(jetMassTrainingDataSig[i*batchSize:(i+1)*batchSize]).cuda()
        #targetv = torch.FloatTensor(trainingLabels[i*batchSize:(i+1)*batchSize]).cuda()
        
        # Barlow Loss
        outSig = gnn(trainingvSig)
        outBkg = gnn(trainingvBkg)
        print(outSig[:5])
        print(outBkg[:5])
        lossClr = clr_criterion(outSig, outBkg)
        #print(trainingvSig[1])
        #print(trainingvBkg[1])
        #print(trainingvMass[1])
        # Correlation Loss
        
        lossCorr1 = cor_criterion(outSig[:,1], trainingvMass)
        lossCorr2 = acr_criterion(trainingvMass, outSig[:,0])
        l = lossClr + lossCorr1 + lossCorr2 
        print(lossClr)
        print(lossCorr1)
        print(lossCorr2)
        #print(outSig)
        #print(trainingvMass)
        
        # Classical BCE loss
        #trainingv = torch.FloatTensor(particleTrainingDataSig[i*batchSize:(i+1)*batchSize]).cuda()
        #out = gnn(trainingv)
        #l = loss(out, targetv)
        
        
        loss_training.append(l.item())
        l.backward()
        optimizer.step()
        loss_string = "Loss: %s" % "{0:.5f}".format(l.item())
        del trainingvSig, trainingvBkg, trainingvMass, l, outSig, outBkg
        torch.cuda.empty_cache()
                   
    toc = time.perf_counter()
    print(f"Training done in {toc - tic:0.4f} seconds")
    tic = time.perf_counter()

    for i in range(int(0.8*datapoints/batchSize)): 
        
        trainingvSig_val = torch.FloatTensor(particleValidationDataSig[i*batchSize:(i+1)*batchSize]).cuda()
        trainingvBkg_val = torch.FloatTensor(particleValidationDataBkg[i*batchSize:(i+1)*batchSize]).cuda()
        trainingvMass_val = torch.FloatTensor(jetMassValidationDataSig[i*batchSize:(i+1)*batchSize]).cuda()
        targetv_val = torch.FloatTensor(validationLabels[i*batchSize:(i+1)*batchSize]).cuda()
        
        
        # Barlow Loss
        outSig_val = gnn(trainingvSig_val)
        outBkg_val = gnn(trainingvBkg_val)
        lossClr = clr_criterion(outSig_val, outBkg_val)
        
        # Correlation Loss
        lossCorr1 = 10*cor_criterion(outSig_val[:,0], trainingvMass_val)
        lossCorr2 = 10*acr_criterion(trainingvMass_val, outSig_val[:,1])
        l_val = lossClr + lossCorr1 + lossCorr2 
        
        
        # Classical validation
        trainingv_val = torch.FloatTensor(particleValidationData[i*batchSize:(i+1)*batchSize]).cuda()
        
        out = gnn(trainingv_val)
        # l_val = loss(out, targetv_val)
        lst.append(softmax(out).cpu().data.numpy())
        loss_val.append(l_val.item())
        correct.append(targetv_val.cpu())
        
        del trainingvSig_val, trainingvBkg_val, trainingvMass_val, targetv_val
    #targetv_cpu = targetv.cpu().data.numpy()
    
    toc = time.perf_counter()
    print(f"Evaluation done in {toc - tic:0.4f} seconds")
    l_val = np.mean(np.array(loss_val))

    predicted = np.concatenate(lst) #(torch.FloatTensor(np.concatenate(lst))).to(device)
    print('\nValidation Loss: ', l_val)

    l_training = np.mean(np.array(loss_training))
    print('Training Loss: ', l_training)
    val_targetv = np.concatenate(correct) #torch.FloatTensor(np.array(correct)).cuda()

    torch.save(gnn.state_dict(), '%s/gnn_%s_last.pth'%(outdir,label))
    if l_val < l_val_best:
        print("new best model")
        l_val_best = l_val
        torch.save(gnn.state_dict(), '%s/gnn_%s_best.pth'%(outdir,label))

    print(val_targetv.shape, predicted.shape)
    print(val_targetv, predicted)
    acc_vals_validation[m] = accuracy_score(val_targetv[:,0],predicted[:,0]>0.5)
    print("Validation Accuracy: ", acc_vals_validation[m])
    loss_vals_training[m] = l_training
    loss_vals_validation[m] = l_val
    loss_std_validation[m] = np.std(np.array(loss_val))
    loss_std_training[m] = np.std(np.array(loss_training))
    if m > 8 and all(loss_vals_validation[max(0, m - 8):m] > min(np.append(loss_vals_validation[0:max(0, m - 8)], 200))):
        print('Early Stopping...')
        print(loss_vals_training, '\n', np.diff(loss_vals_training))
        break
    print()

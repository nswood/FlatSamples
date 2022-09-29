# Imports basics
import os
import numpy as np
import h5py
import json
import setGPU
import sklearn
import corner
import scipy
import time
from tqdm import tqdm 
import utils #import *
import sys
import models
import losses
# Imports neural net tools
import itertools
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import torch.nn.functional as F
#from fast_soft_sort.pytorch_ops import soft_rank
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score,  auc
from torchmetrics import Accuracy
from torchsummary import summary
from sklearn.preprocessing import OneHotEncoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)
import argparse
parser = argparse.ArgumentParser(description='Test.')                                 
parser.add_argument('--opath', action='store', type=str, help='Path to output files.') 
parser.add_argument('--mpath', action='store', type=str, help='Path to model for inference+plotting.') 
parser.add_argument('--plot_text', action='store', type=str, help='Text to add to plot (: delimited).') 
parser.add_argument('--plot_features', action='store_true', default=False, help='Flag to plot features.') 
parser.add_argument('--is_binary', action='store_true', default=False, help='Train only Z\'(inclusive) vs QCD.') 
parser.add_argument('--no_heavy_flavorQCD', action='store_true', default=False, help='Exclude heavy flavor QCD.') 
parser.add_argument('--one_hot_encode_pdgId', action='store_true', default=False, help='One-hot-encode particle pdgId.') 

args = parser.parse_args()
if args.opath:
    os.system("mkdir -p "+args.opath)

srcDir = '/work/tier3/jkrupa/FlatSamples/' 

n_particle_features = 6
n_particles = 100
n_vertex_features = 13
n_vertex = 5

# convert training file
data = h5py.File(os.path.join(srcDir, 'total_df.h5'),'r')
particleData  = utils.reshape_inputs(data['p_features'], n_particle_features)
vertexData    = utils.reshape_inputs(data['SV_features'], n_vertex_features)
singletonData = np.array(data['singletons'])
labels        = singletonData[:,-3:]
singletonFeatureData = np.array(data['singleton_features'])

n_parts = np.count_nonzero(particleData[:,:,0],axis=1)
n_parts = np.expand_dims(n_parts,axis=-1)
singletonData = np.c_[singletonData,n_parts]

if args.is_binary:
   print("Run binary training Z' vs QCD")
   labels = np.expand_dims(np.sum(labels, axis=1),axis=-1)

else:
   print("Running categorical training Z'bb vs cc vs qq vs QCD")

labels[labels>0] = 1

qcd_label = np.zeros((len(labels),1))
is_qcd    = np.where(np.sum(labels,axis=1)==0)
qcd_label[is_qcd] = 1.
labels    = np.concatenate((labels,qcd_label),axis=1)
labels    = labels.astype(np.int)

if args.plot_features:
    print("Plotting all features. This might take a few minutes")
    utils.plot_features(singletonData,labels,utils._singleton_labels,args.opath)
    utils.plot_features(vertexData,labels,utils._SV_features_labels,args.opath,"SV")
    utils.plot_features(particleData,labels,utils._p_features_labels,args.opath,"Particle")
    utils.plot_features(singletonFeatureData,labels,utils._singleton_features_labels,args.opath)

p = np.random.permutation(particleData.shape[0])
particleData, vertexData, singletonData, singletonFeatureData, labels = particleData[p,:,:], vertexData[p,:,:], singletonData[p,:], singletonFeatureData[p,:], labels[p,:]

pdgIdIdx = -1
pdgIdColumn = abs(particleData[:,:,pdgIdIdx])

particleData = particleData[:,:n_particles,:5]

print("Nans: ",(np.argwhere(np.isnan(particleData)),np.argwhere(np.isnan(vertexData))))

if args.one_hot_encode_pdgId:
    print("one hot encoding particle pdgId")
    enc = OneHotEncoder()
    _pdgIds = np.array([[0],[1],[2],[22],[211],[130],[11],[13]])
    enc.fit(_pdgIds)
    encodedParticleData = []
    a = abs(particleData[:,:,-1]).reshape(-1,1)
    b = enc.transform(a).toarray()
    b = b.reshape(labels.shape[0],n_particles,len(_pdgIds))
    particleData = np.concatenate((particleData[:,:,:-1],b),axis=-1)

if args.no_heavy_flavorQCD:
    print("Removing QCD jets with heavy flavor gen hadrons")
    col_idx = utils._singleton_labels.index("zpr_genAK8Jet_hadronFlavour")
    mask = (~labels[:,-1].astype(bool)) | ((labels[:,-1].astype(bool)) & (singletonData[:,col_idx]==0).astype(bool))
    particleData = particleData[mask]
    vertexData = vertexData[mask]
    singletonData = singletonData[mask]
    labels = labels[mask]
    singletonFeatureData = singletonFeatureData[mask]

particleDataTrain,  particleDataVal,  particleDataTest  = utils.train_val_test_split(particleData)
vertexDataTrain,    vertexDataVal,    vertexDataTest    = utils.train_val_test_split(vertexData)
singletonDataTrain, singletonDataVal, singletonDataTest = utils.train_val_test_split(singletonData)
labelsTrain, labelsVal, labelsTest                      = utils.train_val_test_split(labels)
singletonFeatureDataTrain, singletonFeatureDataVal, singletonFeatureDataTest = utils.train_val_test_split(singletonFeatureData)

def run_inference(opath, plot_text, modelName, figpath, model, particleDataTest, labelsTest, singletonDataTest):    
    model.load_state_dict(torch.load(args.mpath))
    eval_classifier(model, plot_text, modelName, figpath, particleDataTest, labelsTest, singletonDataTest)
    return

def train_classifier(classifier, loss, batchSize, nepochs, modelName, outdir,
    particleTrainingData, particleValidationData,  trainingLabels, validationLabels,
    jetMassTrainingData=None, jetMassValidationData=None,
    encoder=None,n_Dim=None, CorrDim=None, 
    svTrainingData=None, svValidationData=None, ):

    optimizer = optim.Adam(classifier.parameters(), lr = 0.001)

    loss_vals_training = np.zeros(nepochs)
    loss_vals_validation = np.zeros(nepochs)
    acc_vals_training = np.zeros(nepochs)
    acc_vals_validation = np.zeros(nepochs)   

    accuracy = Accuracy().cuda()
    model_dir = outdir
    os.system("mkdir -p "+model_dir)
    
    if jetMassTrainingData is not None:
        mass_hist = torch.histc(torch.FloatTensor(jetMassTrainingData), bins=20, min=30., max=400.)
        mass_hist = mass_hist.to(torch.int32)
        one_hots = torch.eye(len(mass_hist))
        one_hots = torch.repeat_interleave(one_hots, mass_hist, dim=0)
        print(one_hots)
        print(one_hots.shape)
    #sys.exit(1)
    for iepoch in range(nepochs):
        loss_training, acc_training = [], []
        loss_validation, acc_validation = [], []

        for istep in tqdm(range(int(len(particleTrainingData)/batchSize))):
            #if istep>10 : continue 
            optimizer.zero_grad()
            batchInputs = torch.FloatTensor(particleTrainingData[istep*batchSize:(istep+1)*batchSize]).cuda()
            batchLabels = torch.FloatTensor(trainingLabels[istep*batchSize:(istep+1)*batchSize]).cuda()
            mass = torch.FloatTensor(jetMassTrainingData[istep*batchSize:(istep+1)*batchSize]).cuda()

            output = classifier(batchInputs)
            l = loss(output, batchLabels, torch.IntTensor(trainingLabels).cuda(), one_hots.cuda(), 100.)
            #print(output) 
            loss_training.append(l.item())
            acc_training.append(accuracy(output,torch.argmax(batchLabels.squeeze(), dim=1)).cpu().detach().numpy())

            l.backward()
            optimizer.step()


            loss_string = "Loss: %s" % "{0:.5f}".format(l.item())
            del batchInputs, batchLabels
            torch.cuda.empty_cache()
            #break
        #sys.exit(1)
        for istep in tqdm(range(int(len(particleValidationData)/batchSize))): 
            #if istep>10 : continue 
            valInputs = torch.FloatTensor(particleValidationData[istep*batchSize:(istep+1)*batchSize]).cuda()
            valLabels = torch.FloatTensor(validationLabels[istep*batchSize:(istep+1)*batchSize]).cuda()

            output = classifier(valInputs)
            l_val  = loss(output, valLabels, torch.IntTensor(trainingLabels).cuda(), one_hots.cuda(), 100.)
 
            loss_validation.append(l_val.item())
            acc_validation.append(accuracy(output,torch.argmax(valLabels.squeeze(), dim=1)).cpu().detach().numpy())

            loss_string = "Loss: %s" % "{0:.5f}".format(l.item())
            del valInputs, valLabels
            torch.cuda.empty_cache()
            #break

        epoch_val_loss = np.mean(loss_validation)
        epoch_val_acc  = np.mean(acc_validation)
        epoch_train_loss = np.mean(loss_training)
        epoch_train_acc  = np.mean(acc_training)

        print("Epoch %i of %i"%(iepoch+1,nepochs))
        print("\tTraining:\tloss=%.5f, acc=%.4f"%(epoch_train_loss, epoch_train_acc))
        print("\tValidation:\tloss=%.5f, acc=%.4f"%(epoch_val_loss, epoch_val_acc))

        loss_vals_validation[iepoch] = epoch_val_loss
        acc_vals_validation[iepoch]  = epoch_val_acc
        loss_vals_training[iepoch] = epoch_train_loss
        acc_vals_training[iepoch]  = epoch_train_acc

        torch.save(classifier.state_dict(), 
            "{}/epoch_{}_{}_loss_{}_{}_acc_{}_{}.pth".format(model_dir,iepoch,modelName.replace(' ','_'),round(loss_vals_training[iepoch],4),round(loss_vals_validation[iepoch],4),round(acc_vals_training[iepoch],4),round(acc_vals_validation[iepoch],4))
        )

        #del valInputs, valLabels
        torch.cuda.empty_cache()


        epoch_patience = 10
        if iepoch > epoch_patience and all(loss_vals_validation[max(0, iepoch - epoch_patience):iepoch] > min(np.append(loss_vals_validation[0:max(0, iepoch - epoch_patience)], 200))):
            print('Early Stopping...')

            utils.plot_loss(loss_vals_training,loss_vals_validation,model_dir)
            break
        elif iepoch == nepochs-1:
            utils.plot_loss(loss_vals_training,loss_vals_validation,model_dir)

    return classifier

def eval_classifier(classifier, training_text, modelName, outdir, 
                    particleTestingData, testingLabels, testingSingletons,
                    encoder=None):
  
    #classifier.eval()  
    with torch.no_grad():
        print("Running predictions on test data")
        predictions = []
        for subtensor in np.array_split(particleTestingData,1000):
            testInputs = torch.FloatTensor(subtensor).cuda()
            predictions.append(classifier(testInputs).cpu().detach().numpy())
            del testInputs
    predictions = [item for sublist in predictions for item in sublist]
    predictions = np.array(predictions).astype(np.float32)
    os.system("mkdir -p "+outdir)
    utils.plot_roc_curve(testingLabels, predictions, training_text, outdir, modelName)

    qcd_idxs = testingLabels[:,-1].astype(bool)
    utils.sculpting_curves(predictions[qcd_idxs,-1], testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="QCD")


    if args.is_binary:
        prob_2prong = predictions[qcd_idxs,0]
        utils.sculpting_curves(prob_2prong, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="2prong")

    else:
        prob_bb = predictions[qcd_idxs,0]
        utils.sculpting_curves(prob_bb, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="bb")
        prob_cc = predictions[qcd_idxs,1]
        utils.sculpting_curves(prob_cc, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="bb")
        prob_qq = predictions[qcd_idxs,2]
        utils.sculpting_curves(prob_qq, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="bb")


if labelsTrain.shape[1] == 2:
    loss = nn.BCELoss(reduction='mean') 
elif labelsTrain.shape[1] == 4:
    loss = losses.adversarial #losses.all_vs_QCD #nn.CrossEntropyLoss()

else:
    raise ValueError("Don't understand shape")

DNN=False
if DNN:
    model = models.DNN(particleDataTrain.shape[1]*particleDataTrain.shape[2],labelsTrain.shape[1]).to(device)
    modelName = "DNN_test" 
    outdir = "/{}/{}/".format(args.opath,modelName.replace(' ','_'))
    outdir = utils.makedir(outdir)

    if args.mpath:
        run_inference(args.mpath, args.plot_text, modelName, args.mpath+"_plots",
                      model, particleDataTest, labelsTest, singletonDataTest)

    else: 
        model = train_classifier(model, loss, 4000,5, modelName, outdir+"/models/", particleDataTrain, particleDataVal, labelsTrain, labelsVal,jetMassTrainingData=singletonDataTrain[:,0] )
        eval_classifier(model, args.plot_text, modelName, outdir+"/plots/", particleDataTest, labelsTest, singletonDataTest,) 

PN=False
if PN: 
    model = models.GraphNetnoSV(n_particles,labelsTrain.shape[1],5,18,softmax=True).to(device)
    modelName = "IN_test"
    outdir = "/{}/{}/".format(args.opath,modelName.replace(' ','_'))
    utils.makedir(outdir)

    #somehow the (parts,features) axes get flipped in the IN 
    particleDataTrain = np.swapaxes(particleDataTrain,1,2)
    particleDataVal = np.swapaxes(particleDataVal,1,2)
    particleDataTest = np.swapaxes(particleDataTest,1,2)

    
    if args.mpath:
        run_inference(args.mpath, args.plot_text, modelName, args.mpath+"_plots", 
                      model, particleDataTest, labelsTest, singletonDataTest)
    else: 
        model = train_classifier(model, loss, 1000,10, modelName, outdir+"/models/", 
                                 particleDataTrain, particleDataVal, labelsTrain, labelsVal)
        eval_classifier(model, args.plot_text, modelName, outdir+"/plots/", particleDataTest, labelsTest, singletonDataTest) 


IN=True
if IN: 
    model = models.GraphNetv2(n_particles,labelsTrain.shape[1],5,hidden=40,De=80,Do=20,softmax=True).to(device)
    print(model)
    modelName = "IN_test"
    outdir = "/{}/{}/".format(args.opath,modelName.replace(' ','_'))
    outdir = utils.makedir(outdir)

    #somehow the (parts,features) axes get flipped in the IN 
    particleDataTrain = np.swapaxes(particleDataTrain,1,2)
    particleDataVal = np.swapaxes(particleDataVal,1,2)
    particleDataTest = np.swapaxes(particleDataTest,1,2)
    #print(particleDataTrain[0])
    #print(particleDataTrain[1])
 
    print(particleDataTrain.shape)
    #particleDataTrain = particleDataTrain.
    model = torch.nn.DataParallel(model)
    
    if args.mpath:
        run_inference(args.mpath, args.plot_text, modelName, args.mpath+"_plots", 
                      model, particleDataTest, labelsTest, singletonDataTest)
    else: 
        model = train_classifier(model, loss, 1028,5, modelName, outdir+"/models/", 
                                 particleDataTrain, particleDataVal, labelsTrain, labelsVal, jetMassTrainingData=singletonDataTrain[:,0])
        eval_classifier(model, args.plot_text, modelName, outdir+"/plots/", particleDataTest, labelsTest, singletonDataTest) 




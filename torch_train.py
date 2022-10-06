# Imports basics
import os
import numpy as np
import h5py
import json
#import setGPU
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
parser.add_argument('--loss', action='store', type=str, help='Name of loss to use.') 
parser.add_argument('--opath', action='store', type=str, help='Path to output files.') 
parser.add_argument('--mpath', action='store', type=str, help='Path to model for inference+plotting.') 
parser.add_argument('--nparts', action='store', type=int,default=50, help='Number of particles.') 
parser.add_argument('--nepochs', action='store', type=int,default=10, help='Number of training epochs.') 
parser.add_argument('--LAMBDA_ADV', action='store', type=float,default=None, help='Adversarial lambda.') 
parser.add_argument('--plot_text', action='store', type=str, help='Text to add to plot (: delimited).') 
parser.add_argument('--plot_features', action='store_true', default=False, help='Flag to plot features.') 
parser.add_argument('--is_binary', action='store_true', default=False, help='Train only Z\'(inclusive) vs QCD.') 
parser.add_argument('--no_heavy_flavorQCD', action='store_true', default=False, help='Exclude heavy flavor QCD.') 
parser.add_argument('--one_hot_encode_pdgId', action='store_true', default=False, help='One-hot-encode particle pdgId.') 

args = parser.parse_args()

assert(args.loss)
   
if args.opath:
    os.system("mkdir -p "+args.opath)


srcDir = '/work/tier3/jkrupa/FlatSamples/' 

n_particle_features = 6
n_particles = args.nparts
n_vertex_features = 13
n_vertex = 5
batchsize = 2000
n_epochs = args.nepochs

print("Running with %i particle features, %i particles, %i vertex features, %i vertices, %i batchsize, %i epochs"%(n_particle_features,n_particles,n_vertex_features,n_vertex,batchsize,n_epochs))

if args.loss == 'bce':
    loss = nn.BCELoss()
elif args.loss == 'all_vs_qcd':
    loss = losses.all_vs_QCD
elif args.loss == 'jsd':
    loss = losses.jsd
elif args.loss == 'disco':
    loss = losses.disco
else:
    raise NameError("Don't understand loss")

if (args.loss == 'jsd' or args.loss == 'disco') and not args.LAMBDA_ADV:
    raise ValueError("must provide lambda_adv for adversarial")
# convert training file
data = h5py.File(os.path.join(srcDir, 'total_df.h5'),'r')
print(data['SV_features'].shape)
particleData  = utils.reshape_inputs(data['p_features'], n_particle_features)
vertexData    = utils.reshape_inputs(data['SV_features'], n_vertex_features)
print(vertexData.shape)
singletonData = np.array(data['singletons'])
labels        = singletonData[:,-3:]
singletonFeatureData = np.array(data['singleton_features'])

#print("Nans:",np.where(particleData==np.nan)[0], np.where(vertexData==np.nan)[0], np.where(singletonData==np.nan)[0], np.where(labels==np.nan)[0], np.where(singletonFeatureData==np.nan)[0])
#print("Infs:",np.where(particleData==np.inf)[0], np.where(vertexData==np.inf)[0], np.where(singletonData==np.inf)[0], np.where(labels==np.inf)[0], np.where(singletonFeatureData==np.inf)[0])
#print("negInfs:",np.where(particleData==np.isneginf)[0], np.where(vertexData==np.isneginf)[0], np.where(singletonData==np.isneginf)[0], np.where(labels==np.isneginf)[0], np.where(singletonFeatureData==np.isneginf)[0])

#nans = [item for sublist in (np.where(particleData==np.nan)[0] , np.where(vertexData==np.nan)[0] , np.where(singletonData==np.nan)[0] ,  np.where(labels==np.nan)[0] , np.where(singletonFeatureData==np.nan)[0] , np.where(particleData==np.inf)[0] , np.where(vertexData==np.inf)[0] , np.where(singletonData==np.inf)[0] ,  np.where(labels==np.inf)[0] , np.where(singletonFeatureData==np.inf)[0]) for item in sublist]
#print(nans) 

x = np.where(~np.isfinite(vertexData).all(axis=1))
print("not finite: ",x[0])

nan_mask = np.ones(len(particleData),dtype=bool)
nan_mask[x[0]] = False 
particleData = particleData[nan_mask]
vertexData = vertexData[nan_mask]
singletonData = singletonData[nan_mask]
labels = labels[nan_mask]
singletonFeatureData = singletonFeatureData[nan_mask]

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

particleData = particleData[:,:n_particles,:n_particle_features]

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

    accuracy = Accuracy().to(device)
    model_dir = outdir
    os.system("mkdir -p "+model_dir)
    n_massbins = 20
    if jetMassTrainingData is not None:
        bins = torch.linspace(30.,400.,n_massbins)
        one_hots = torch.bucketize(torch.FloatTensor(jetMassTrainingData),bins)
       
    for iepoch in range(nepochs):
        loss_training, acc_training = [], []
        loss_validation, acc_validation = [], []

        for istep in tqdm(range(int(len(particleTrainingData)/batchSize))):
            #if istep>10 : continue 
            optimizer.zero_grad()
            batchInputs = torch.FloatTensor(particleTrainingData[istep*batchSize:(istep+1)*batchSize]).to(device)
            batchLabels = torch.FloatTensor(trainingLabels[istep*batchSize:(istep+1)*batchSize]).to(device)
            mass = torch.FloatTensor(jetMassTrainingData[istep*batchSize:(istep+1)*batchSize]).to(device)

            if svTrainingData is not None:
                batchInputsSV = torch.FloatTensor(svTrainingData[istep*batchSize:(istep+1)*batchSize]).to(device)
                output = classifier(batchInputs, batchInputsSV)
            else:
                output = classifier(batchInputs)

            if args.loss == 'jsd':
                l = loss(output, batchLabels, one_hots[istep*batchSize:(istep+1)*batchSize].to(device), n_massbins=n_massbins, LAMBDA_ADV=args.LAMBDA_ADV)
            elif args.loss == 'disco':
                l = loss(output, batchLabels, mass, LAMBDA_ADV=args.LAMBDA_ADV)
            else:
                l = torch.nn.functional.binary_cross_entropy(output, batchLabels)

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
            valInputs = torch.FloatTensor(particleValidationData[istep*batchSize:(istep+1)*batchSize]).to(device)
            valLabels = torch.FloatTensor(validationLabels[istep*batchSize:(istep+1)*batchSize]).to(device)
            mass = torch.FloatTensor(jetMassValidationData[istep*batchSize:(istep+1)*batchSize]).to(device)

            if svTrainingData is not None:
                valInputsSV = torch.FloatTensor(svValidationData[istep*batchSize:(istep+1)*batchSize]).to(device)
                output = classifier(valInputs, valInputsSV)
            else:
                output = classifier(valInputs)
            if args.loss == 'jsd':
                l_val  = loss(output, valLabels, one_hots[istep*batchSize:(istep+1)*batchSize].to(device), n_massbins=n_massbins, LAMBDA_ADV=args.LAMBDA_ADV)
            elif args.loss == 'disco':
                l = loss(output, valLabels, mass, LAMBDA_ADV=args.LAMBDA_ADV)
            else: 
                l_val = torch.nn.functional.binary_cross_entropy(output, valLabels)
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
                    svTestingData=None,encoder=None):
  
    #classifier.eval()  
    with torch.no_grad():
        print("Running predictions on test data")
        predictions = []
  
        if svTestingData is not None:
            for subtensor,subtensorSV in zip(np.array_split(particleTestingData,1000),np.array_split(svTestingData,1000)):
                testInputs = torch.FloatTensor(subtensor).to(device)
                testInputsSV = torch.FloatTensor(subtensorSV).to(device)
                predictions.append(classifier(testInputs,testInputsSV).cpu().detach().numpy())
                del testInputs
       
        else:
            for subtensor in np.array_split(particleTestingData,1000):
                testInputs = torch.FloatTensor(subtensor).to(device)
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
        utils.sculpting_curves(prob_2prong, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="Z\'")

    else:
        prob_bb = predictions[qcd_idxs,0]
        utils.sculpting_curves(prob_bb, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="bb")
        prob_cc = predictions[qcd_idxs,1]
        utils.sculpting_curves(prob_cc, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="bb")
        prob_qq = predictions[qcd_idxs,2]
        utils.sculpting_curves(prob_qq, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="bb")




DNN=0
if DNN:
    model = models.DNN(particleDataTrain.shape[1]*particleDataTrain.shape[2],labelsTrain.shape[1]).to(device)
    modelName = "DNN_test" 
    outdir = "/{}/{}/".format(args.opath,modelName.replace(' ','_'))
    outdir = utils.makedir(outdir)

    if args.mpath:
        run_inference(args.mpath, args.plot_text, modelName, args.mpath+"_plots",
                      model, particleDataTest, labelsTest, singletonDataTest)

    else: 
        model = train_classifier(model, loss, 4000,5, modelName, outdir+"/models/", particleDataTrain, particleDataVal, labelsTrain, labelsVal,jetMassTrainingData=singletonDataTrain[:,0], jetMassValidationData=singletonDataVal[:,0] )
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

    print(particleDataTrain[0]) 
    print(particleDataTrain.shape) 
    if args.mpath:
        run_inference(args.mpath, args.plot_text, modelName, args.mpath+"_plots", 
                      model, particleDataTest, labelsTest, singletonDataTest)
    else: 
        model = train_classifier(model, loss, 1000,10, modelName, outdir+"/models/", 
                                 particleDataTrain, particleDataVal, labelsTrain, labelsVal)
        eval_classifier(model, args.plot_text, modelName, outdir+"/plots/", particleDataTest, labelsTest, singletonDataTest) 


IN=1
if IN: 
    model = models.GraphNetv2(n_particles,labelsTrain.shape[1],6,n_vertices=5, params_v=13, pv_branch=True, hidden=20, De=20, Do=20,softmax=True)
    #model = models.GraphNetv2(n_particles,labelsTrain.shape[1],5,hidden=40,De=40,Do=30,softmax=True)
    model = model.to(device)
    modelName = "IN_test"
    outdir = "/{}/{}/".format(args.opath,modelName.replace(' ','_'))
    outdir = utils.makedir(outdir)

    #somehow the (parts,features) axes get flipped in the IN 
    particleDataTrain = np.swapaxes(particleDataTrain,1,2)
    particleDataVal = np.swapaxes(particleDataVal,1,2)
    particleDataTest = np.swapaxes(particleDataTest,1,2)
    vertexDataTrain = np.swapaxes(vertexDataTrain,1,2)
    vertexDataVal = np.swapaxes(vertexDataVal,1,2)
    vertexDataTest = np.swapaxes(vertexDataTest,1,2)

    model = torch.nn.DataParallel(model)
    
    if args.mpath:
        run_inference(args.mpath, args.plot_text, modelName, args.mpath+"_plots", 
                      model, particleDataTest, labelsTest, singletonDataTest)
    else: 
        model = train_classifier(model, loss, 1024,20, modelName, outdir+"/models/", 
                                 particleDataTrain, particleDataVal, labelsTrain, labelsVal, jetMassTrainingData=singletonDataTrain[:,0],
                                 svTrainingData=vertexDataTrain, svValidationData=vertexDataVal,)
        eval_classifier(model, args.plot_text, modelName, outdir+"/plots/", particleDataTest, labelsTest, singletonDataTest, svTestingData=vertexDataTest) 




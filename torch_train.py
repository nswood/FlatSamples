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
parser.add_argument('--bce_disco', action='store', type=str, default=None, help='use BCE with disco.') 
parser.add_argument('--opath', action='store', type=str, help='Path to output files.') 
parser.add_argument('--mpath', action='store', type=str, help='Path to model for inference+plotting.') 
parser.add_argument('--De', action='store', type=int,default=20, help='Output dimension.') 
parser.add_argument('--Do', action='store', type=int,default=20, help='Output dimension.') 
parser.add_argument('--hidden', action='store', type=int,default=20, help='Hidden dimension size.') 
parser.add_argument('--nparts', action='store', type=int,default=50, help='Number of particles.') 
parser.add_argument('--nepochs', action='store', type=int,default=10, help='Number of training epochs.') 
parser.add_argument('--batchsize', action='store', type=int,default=2000, help='Number of training epochs.') 
parser.add_argument('--LAMBDA_ADV', action='store', type=float,default=None, help='Adversarial lambda.') 
parser.add_argument('--plot_text', action='store', type=str, help='Text to add to plot (: delimited).') 
parser.add_argument('--plot_features', action='store_true', default=False, help='Flag to plot features.') 
parser.add_argument('--run_captum', action='store_true', default=False, help='Flat to run attribution test') 
parser.add_argument('--test_run', action='store_true', default=False, help='Short run for testing.') 
parser.add_argument('--make_PN', action='store_true', default=False, help='Flag to make PN plots.') 
parser.add_argument('--is_binary', action='store_true', default=False, help='Train only Z\'(inclusive) vs QCD.') 
parser.add_argument('--no_heavy_flavorQCD', action='store_true', default=False, help='Exclude heavy flavor QCD.') 
parser.add_argument('--one_hot_encode_pdgId', action='store_true', default=False, help='One-hot-encode particle pdgId.') 
parser.add_argument('--SV', action='store_true', default=False, help='Run with SVs.') 
parser.add_argument('--event', action='store_true', default=False, help='Run with event info.') 

args = parser.parse_args()

if not args.make_PN:
    assert(args.loss)

if "disco" in args.loss and args.bce_disco is None:
    raise ValueError("Need BCE or categorical for disco")

if args.opath:
    os.system("mkdir -p "+args.opath)


srcDir = '/work/tier3/jkrupa/FlatSamples/' 

n_particle_features = 6
n_particles = args.nparts
n_vertex_features = 13
n_vertex = 5
batchsize = args.batchsize
n_epochs = args.nepochs

print("Running with %i particle features, %i particles, %i vertex features, %i vertices, %i batchsize, %i epochs"%(n_particle_features,n_particles,n_vertex_features,n_vertex,batchsize,n_epochs))
_sigmoid=False
_softmax=True
if args.loss == 'bce':
    loss = nn.BCELoss()
elif args.loss == 'all_vs_QCD':
    loss = losses.all_vs_QCD
    _sigmoid=True
    _softmax=False
elif args.loss == 'jsd':
    loss = losses.jsd
elif args.loss == 'disco':
    loss = losses.disco
elif args.loss == 'disco_all_vs_QCD':
    loss = losses.disco_all_vs_QCD
    _sigmoid=True
    _softmax=False
else:
    raise NameError("Don't understand loss")

if ('jsd' in args.loss or 'disco' in args.loss) and not args.LAMBDA_ADV:
    raise ValueError("must provide lambda_adv for adversarial")
# convert training file
data = h5py.File(os.path.join(srcDir, 'total_df.h5'),'r')
particleData  = utils.reshape_inputs(data['p_features'], n_particle_features)
vertexData    = utils.reshape_inputs(data['SV_features'], n_vertex_features)
singletonData = np.array(data['singletons'])
labels        = singletonData[:,-3:]
singletonFeatureData = np.array(data['singleton_features'])



if True:

    x = np.where(~np.isfinite(vertexData).all(axis=1))
    y = np.where(~np.isfinite(particleData).all(axis=1))
    z = np.where(~np.isfinite(singletonFeatureData).all(axis=1))
    print(x)
    print(y)
    print(z)
    nan_mask = np.ones(len(particleData),dtype=bool)
    nan_mask[x[0]] = False 
    nan_mask[y[0]] = False 
    nan_mask[z[0]] = False 
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

def run_inference(opath, plot_text, modelName, figpath, model, particleDataTest, labelsTest, singletonDataTest, svTestingData=None, eventTestingData=None):    
    model.load_state_dict(torch.load(args.mpath))
    eval_classifier(model, plot_text, modelName, figpath, particleDataTest, labelsTest, singletonDataTest,svTestingData=vertexDataTest,eventTestingData=eventTestingData)
    return

def train_classifier(classifier, loss, batchSize, nepochs, modelName, outdir,
    particleTrainingData, particleValidationData,  trainingLabels, validationLabels,
    jetMassTrainingData=None, jetMassValidationData=None,
    encoder=None,n_Dim=None, CorrDim=None, 
    svTrainingData=None, svValidationData=None,
    eventTrainingData=None, eventValidationData=None, 
    ):

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
            if (args.test_run and istep>10 ): break
            optimizer.zero_grad()
            batchInputs = torch.FloatTensor(particleTrainingData[istep*batchSize:(istep+1)*batchSize]).to(device)
            batchLabels = torch.FloatTensor(trainingLabels[istep*batchSize:(istep+1)*batchSize]).to(device)
            mass = torch.FloatTensor(jetMassTrainingData[istep*batchSize:(istep+1)*batchSize]).to(device)

            if svTrainingData is not None and eventTrainingData is not None:
                batchInputsSV = torch.FloatTensor(svTrainingData[istep*batchSize:(istep+1)*batchSize]).to(device)
                batchInputsE = torch.FloatTensor(eventTrainingData[istep*batchSize:(istep+1)*batchSize]).to(device)
                output = classifier(batchInputs, batchInputsSV, batchInputsE)
            elif svTrainingData is not None:
                batchInputsSV = torch.FloatTensor(svTrainingData[istep*batchSize:(istep+1)*batchSize]).to(device)
                output = classifier(batchInputs, batchInputsSV)
            else:
                output = classifier(batchInputs)

            if args.loss == 'jsd':
                l = loss(output, batchLabels, one_hots[istep*batchSize:(istep+1)*batchSize].to(device), n_massbins=n_massbins, LAMBDA_ADV=args.LAMBDA_ADV)
            elif 'disco' in args.loss:
                l = loss(output, batchLabels, mass, args.bce_disco, LAMBDA_ADV=args.LAMBDA_ADV,)
            else:
                l = torch.nn.functional.binary_cross_entropy(output, batchLabels)

            loss_training.append(l.item())
            acc_training.append(accuracy(output,torch.argmax(batchLabels.squeeze(), dim=1)).cpu().detach().numpy())

            l.backward()
            optimizer.step()


            loss_string = "Loss: %s" % "{0:.5f}".format(l.item())
            del batchInputs, batchInputsSV, batchInputsE, batchLabels
            torch.cuda.empty_cache()
            #break
        #sys.exit(1)
        for istep in tqdm(range(int(len(particleValidationData)/batchSize))): 
            if (args.test_run and istep>10 ): break
            valInputs = torch.FloatTensor(particleValidationData[istep*batchSize:(istep+1)*batchSize]).to(device)
            valLabels = torch.FloatTensor(validationLabels[istep*batchSize:(istep+1)*batchSize]).to(device)
            mass = torch.FloatTensor(jetMassValidationData[istep*batchSize:(istep+1)*batchSize]).to(device)

            if svTrainingData is not None and eventTrainingData is not None:
                valInputsSV = torch.FloatTensor(svValidationData[istep*batchSize:(istep+1)*batchSize]).to(device)
                valInputsE = torch.FloatTensor(eventValidationData[istep*batchSize:(istep+1)*batchSize]).to(device)
                output = classifier(valInputs, valInputsSV, valInputsE)
            elif svTrainingData is not None:
                valInputsSV = torch.FloatTensor(svValidationData[istep*batchSize:(istep+1)*batchSize]).to(device)
                output = classifier(valInputs, valInputsSV)
            else:
                output = classifier(valInputs)
            if args.loss == 'jsd':
                l_val  = loss(output, valLabels, one_hots[istep*batchSize:(istep+1)*batchSize].to(device), n_massbins=n_massbins, LAMBDA_ADV=args.LAMBDA_ADV)
            elif 'disco' in args.loss:
                l_val = loss(output, valLabels, mass, args.bce_disco, LAMBDA_ADV=args.LAMBDA_ADV,)
            else: 
                l_val = torch.nn.functional.binary_cross_entropy(output, valLabels)
            loss_validation.append(l_val.item())
            acc_validation.append(accuracy(output,torch.argmax(valLabels.squeeze(), dim=1)).cpu().detach().numpy())

            loss_string = "Loss: %s" % "{0:.5f}".format(l.item())
            del valInputs, valInputsSV, valInputsE, valLabels
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
                    svTestingData=None,eventTestingData=None,encoder=None):
  
    #classifier.eval()  
    with torch.no_grad():
        print("Running predictions on test data")
        predictions = []
 
        if svTestingData is not None and eventTestingData is not None:
            for subtensor,subtensorSV,subtensorE in zip(np.array_split(particleTestingData,1000),np.array_split(svTestingData,1000),np.array_split(eventTestingData,1000)):

                testInputs = torch.FloatTensor(subtensor).to(device)
                testInputsSV = torch.FloatTensor(subtensorSV).to(device)
                testInputsE = torch.FloatTensor(subtensorE).to(device)
                predictions.append(classifier(testInputs,testInputsSV,testInputsE).cpu().detach().numpy())
                del testInputs, testInputsSV, testInputsE
                #if (args.test_run): break 
 
        elif svTestingData is not None:
            for subtensor,subtensorSV in zip(np.array_split(particleTestingData,1000),np.array_split(svTestingData,1000)):
             
                testInputs = torch.FloatTensor(subtensor).to(device)
                testInputsSV = torch.FloatTensor(subtensorSV).to(device)
                predictions.append(classifier(testInputs,testInputsSV).cpu().detach().numpy())
                del testInputs, testInputsSV
                #if (args.test_run): break 
        else:
            for subtensor in np.array_split(particleTestingData,1000):
                testInputs = torch.FloatTensor(subtensor).to(device)
                predictions.append(classifier(testInputs).cpu().detach().numpy())
                del testInputs
                #if (args.test_run): break 
    predictions = [item for sublist in predictions for item in sublist]
    predictions = np.array(predictions).astype(np.float32)
    os.system("mkdir -p "+outdir)


    qcd_idxs = testingLabels[:,-1].astype(bool)
    utils.plot_correlation(predictions[qcd_idxs,-1],testingSingletons[qcd_idxs,0], "QCD output score","QCD jet $m_{SD}$ (GeV)", np.linspace(0,1,100),np.linspace(40,350,40),outdir, "qcd_vs_mass")
    utils.plot_roc_curve(testingLabels, predictions, training_text, outdir, modelName)
    utils.sculpting_curves(predictions[qcd_idxs,-1], testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="QCD")

    if args.is_binary:
        
        prob_2prong = predictions[qcd_idxs,0]
        utils.sculpting_curves(prob_2prong, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="Z\'")

    else:
        prob_bb = predictions[qcd_idxs,0]
        utils.sculpting_curves(prob_bb, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="bb")
        prob_cc = predictions[qcd_idxs,1]
        utils.sculpting_curves(prob_cc, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="cc")
        prob_qq = predictions[qcd_idxs,2]
        utils.sculpting_curves(prob_qq, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="qq")

if args.make_PN:
    predictions = singletonData[:,[utils._singleton_labels.index("zpr_fj_particleNetMD_Xbb"), utils._singleton_labels.index("zpr_fj_particleNetMD_Xcc"), utils._singleton_labels.index("zpr_fj_particleNetMD_Xqq"), utils._singleton_labels.index("zpr_fj_particleNetMD_QCD")]]
    utils.plot_roc_curve(labels, predictions, args.plot_text, args.opath, "particleNet-MD")
    
    qcd_idxs = labels[:,-1].astype(bool)
    prob_bb = predictions[qcd_idxs,0]
    utils.sculpting_curves(prob_bb, singletonData[qcd_idxs,:],"ParticleNet-MD:bb score", args.opath, "particleNet-MD")
    prob_cc = predictions[qcd_idxs,1]
    utils.sculpting_curves(prob_cc, singletonData[qcd_idxs,:], "ParticleNet-MD:cc score", args.opath, "particleNet-MD")
    prob_qq = predictions[qcd_idxs,2]
    utils.sculpting_curves(prob_qq, singletonData[qcd_idxs,:], "ParticleNet-MD:qq score", args.opath, "particleNet-MD")
    prob_QCD = predictions[qcd_idxs,3]
    utils.sculpting_curves(prob_QCD, singletonData[qcd_idxs,:], "ParticleNet-MD:QCD score", args.opath, "particleNet-MD")
    sys.exit(1)

DNN=0
PN=0
IN_SV=0
IN_noSV=0
IN_SV_event=1
if DNN:
    model = models.DNN(particleDataTrain.shape[1]*particleDataTrain.shape[2],labelsTrain.shape[1]).to(device)

if PN: 

    model = models.ParticleNetTagger("PN",n_particle_features,n_vertex_features,labelsTrain.shape[1])

if IN_SV_event:
    model = models.GraphNetv2("IN_SV_event",n_particles,labelsTrain.shape[1],6,n_vertices=5,params_v=13,params_e=27,pv_branch=True,event_branch=True, hidden=args.hidden, De=args.De, Do=args.Do,sigmoid=_sigmoid,softmax=_softmax)
    particleDataTrain = np.swapaxes(particleDataTrain,1,2)
    particleDataVal = np.swapaxes(particleDataVal,1,2)
    particleDataTest = np.swapaxes(particleDataTest,1,2)
    vertexDataTrain = np.swapaxes(vertexDataTrain,1,2)
    vertexDataVal = np.swapaxes(vertexDataVal,1,2)
    vertexDataTest = np.swapaxes(vertexDataTest,1,2)
    #singletonFeatureDataTrain = np.swapaxes(singletonFeatureDataTrain,1,2)
    #singletonFeatureDataVal = np.swapaxes(singletonFeatureDataVal,1,2)
    #singletonFeatureDataTest = np.swapaxes(singletonFeatureDataTest,1,2)

if IN_SV: 
    model = models.GraphNetv2("IN_SV",n_particles,labelsTrain.shape[1],6,n_vertices=5, params_v=13, pv_branch=True, hidden=args.hidden, De=args.De, Do=args.Do,sigmoid=_sigmoid,softmax=_softmax)

    #somehow the (parts,features) axes get flipped in the IN 
    particleDataTrain = np.swapaxes(particleDataTrain,1,2)
    particleDataVal = np.swapaxes(particleDataVal,1,2)
    particleDataTest = np.swapaxes(particleDataTest,1,2)
    vertexDataTrain = np.swapaxes(vertexDataTrain,1,2)
    vertexDataVal = np.swapaxes(vertexDataVal,1,2)
    vertexDataTest = np.swapaxes(vertexDataTest,1,2)

if IN_noSV: 
    model = models.GraphNetv2("IN_noSV",n_particles,labelsTrain.shape[1],6,hidden=args.hidden, De=args.De, Do=args.Do,sigmoid=_sigmoid,softmax=_softmax)

    #somehow the (parts,features) axes get flipped in the IN 
    particleDataTrain = np.swapaxes(particleDataTrain,1,2)
    particleDataVal = np.swapaxes(particleDataVal,1,2)
    particleDataTest = np.swapaxes(particleDataTest,1,2)
 
   
if not args.SV:
    vertexDataTrain=None; vertexDataVal=None; vertexDataTest=None; 
if not args.event:
    singletonFeatureDataTrain=None;singletonFeatureDataVal=None;singletonFeatureDataTest=None;

model = model.to(device)
outdir = "/{}/{}/".format(args.opath,model.name.replace(' ','_'))
outdir = utils.makedir(outdir)
 
if args.mpath:
    run_inference(args.mpath, args.plot_text, model.name, args.mpath+"_plots", 
                  model, particleDataTest, labelsTest, singletonDataTest, svTestingData=vertexDataTest, eventTestingData=singletonFeatureDataTest
    )

else: 
    model = train_classifier(model, loss, batchsize, n_epochs, model.name, outdir+"/models/", 
                             particleDataTrain, particleDataVal, labelsTrain, labelsVal, jetMassTrainingData=singletonDataTrain[:,0],
                             jetMassValidationData=singletonDataVal[:,0],
                             svTrainingData=vertexDataTrain, svValidationData=vertexDataVal,
                             eventTrainingData=singletonFeatureDataTrain, eventValidationData=singletonFeatureDataVal,
    )
    #eval_classifier(model, args.plot_text, model.name, outdir+"/plots/", particleDataTest, labelsTest, singletonDataTest, svTestingData=vertexDataTest, eventTestingData=singletonFeatureDataTest,) 

if args.run_captum:
    from captum.attr import IntegratedGradients
    model.eval()
    torch.manual_seed(123)
    np.random.seed(123)
    baseline = torch.zeros(1,particleDataTrain.shape[1],particleDataTrain.shape[2]).to(device)
    baselineSV = torch.zeros(1,vertexDataTrain.shape[1],vertexDataTrain.shape[2]).to(device)
    baselineE  = torch.zeros(1,singletonFeatureDataTrain.shape[1]).to(device)
    inputs = torch.rand(1,particleDataTrain.shape[1],particleDataTrain.shape[2]).to(device)
    inputsSV = torch.rand(1,vertexDataTrain.shape[1],vertexDataTrain.shape[2]).to(device)
    inputsE = torch.rand(1,singletonFeatureDataTrain.shape[1]).to(device)
 
    ig = IntegratedGradients(model)
    os.system("mkdir -p "+outdir+"/captum/")
    if IN_SV_event:
        attributions, delta = ig.attribute((inputs,inputsSV,inputsE,), (baseline,baselineSV,baselineE), target=3, return_convergence_delta=True)
        np.savez(outdir+"/captum/qcd_score.npz",pf=attributions[0].cpu().detach().numpy(),sv=attributions[1].cpu().detach().numpy(),event=attributions[2].cpu().detach().numpy())
        attributions, delta = ig.attribute((inputs,inputsSV,inputsE,), (baseline,baselineSV,baselineE), target=2, return_convergence_delta=True)
        np.savez(outdir+"/captum/qq_score.npz",pf=attributions[0].cpu().detach().numpy(),sv=attributions[1].cpu().detach().numpy(),event=attributions[2].cpu().detach().numpy())
        attributions, delta = ig.attribute((inputs,inputsSV,inputsE,), (baseline,baselineSV,baselineE), target=1, return_convergence_delta=True)
        np.savez(outdir+"/captum/cc_score.npz",pf=attributions[0].cpu().detach().numpy(),sv=attributions[1].cpu().detach().numpy(),event=attributions[2].cpu().detach().numpy())
        attributions, delta = ig.attribute((inputs,inputsSV,inputsE,), (baseline,baselineSV,baselineE), target=0, return_convergence_delta=True)
        np.savez(outdir+"/captum/bb_score.npz",pf=attributions[0].cpu().detach().numpy(),sv=attributions[1].cpu().detach().numpy(),event=attributions[2].cpu().detach().numpy())




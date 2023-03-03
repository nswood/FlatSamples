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

p = utils.ArgumentParser()
p.add_args(
    ('--loss', p.STR), ('--model', p.STR), ('--nepochs', p.INT),
    ('--ipath', p.STR), ('--vpath', p.STR), ('--opath', p.STR),
    ('--mpath', p.STR),
    ('--De', p.FLOAT), ('--Do',p.FLOAT), ('--hidden',p.FLOAT),
    ('--nparts', p.INT),('--LAMBDA_ADV',p.FLOAT), 
    ('--plot_text', p.STR), ('--mini_dataset',p.STORE_TRUE),
    ('--plot_features', p.STORE_TRUE), ('--run_captum',p.STORE_TRUE), ('--test_run',p.STORE_TRUE),
    ('--make_PN', p.STORE_TRUE), ('--make_N2',p.STORE_TRUE), ('--is_binary',p.STORE_TRUE),
    ('--is_peaky',p.STORE_TRUE), ('--no_heavy_flavorQCD',p.STORE_TRUE),
    ('--one_hot_encode_pdgId',p.STORE_TRUE),('--SV',p.STORE_TRUE),
    ('--temperature', p.FLOAT), ('--n_out_nodes',p.INT),
    ('--qcd_only',p.STORE_TRUE), ('--seed_only',p.STORE_TRUE),
    ('--abseta',p.STORE_TRUE), ('--kinematics_only',p.STORE_TRUE),
    ('--istransformer',p.STORE_TRUE),
    ('--num_encoders', p.INT),('--is_decoder',p.STORE_TRUE),
    ('--embedding_size', p.INT), ('--hidden_size', p.INT), ('--feature_size', p.INT),
    ('--num_attention_heads', p.INT), ('--intermediate_size', p.INT),
    ('--label_size', p.INT), ('--num_hidden_layers', p.INT), ('--batchsize', p.INT),
    ('--mask_charged', p.STORE_TRUE), ('--lr', {'type': float}),
    ('--attention_band', p.INT),
    ('--epoch_offset', p.INT),
    ('--from_snapshot'),
    ('--lr_schedule', p.STORE_TRUE), '--plot',
    ('--pt_weight', p.STORE_TRUE), ('--num_max_files', p.INT),
    ('--num_max_particles', p.INT), ('--dr_adj', p.FLOAT),
    ('--beta', p.STORE_TRUE),
    ('--lr_policy'), ('--grad_acc', p.INT),
)
args = p.parse_args()

args.nparts = 100

from dataset_loader_gpu import zpr_loader
data_train = zpr_loader(args.ipath) 
data_val = zpr_loader(args.vpath)

from torch.utils.data import DataLoader
train_loader = DataLoader(data_train, batch_size=args.batchsize,shuffle=False)
val_loader = DataLoader(data_val, batch_size=args.batchsize,shuffle=False)

assert(args.model)
if not args.make_PN:
    assert(args.loss)

if not args.is_binary and args.make_N2:
    raise ValueError("need binary for N2 plots")
if args.opath:
    os.system("mkdir -p ./"+args.opath)


#srcDir = '/work/tier3/jkrupa/FlatSamples/data//30Oct22-MiniAODv2/30Oct22/zpr_fj_msd/2017/' 
#srcDir = '/work/tier3/jkrupa/FlatSamples/data/'
n_particle_features = 6
n_particles = args.nparts
n_vertex_features = 13
n_vertex = 5
batchsize = args.batchsize
n_epochs = args.nepochs

print("Running with %i particle features, %i particles, %i vertex features, %i vertices, %i batchsize, %i epochs"%(n_particle_features,n_particles,n_vertex_features,n_vertex,batchsize,n_epochs))
print("Loss: ", args.loss)

_sigmoid=False
_softmax=True
if args.loss == 'bce':
    loss = nn.BCELoss()
elif args.loss == 'categorical':
    loss = nn.CrossEntropyLoss()
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
#data = h5py.File(os.path.join(srcDir, 'tot.h5'),'r')

if False:
    print("Removing NaNs...")
    x = np.where(~np.isfinite(vertexData).all(axis=1))
    y = np.where(~np.isfinite(particleData).all(axis=1))
    z = np.where(~np.isfinite(singletonFeatureData).all(axis=1))
    nan_mask = np.ones(len(particleData),dtype=bool)
    nan_mask[x[0]] = False 
    nan_mask[y[0]] = False 
    nan_mask[z[0]] = False
    print("# events with NaNs (removed): ", np.sum((nan_mask==False).astype(int))) 
    particleData = particleData[nan_mask]
    vertexData = vertexData[nan_mask]
    singletonData = singletonData[nan_mask]
    labels = labels[nan_mask]
    singletonFeatureData = singletonFeatureData[nan_mask]
    
if 'all_vs_QCD' in args.loss:
    labels = labels[:,:3]      

if args.plot_features:
    print("Plotting all features. This might take a few minutes")
    utils.plot_features(singletonData,labels,utils._singleton_labels,args.opath)
    utils.plot_features(vertexData,labels,utils._SV_features_labels,args.opath,"SV")
    utils.plot_features(particleData,labels,utils._p_features_labels,args.opath,"Particle")
    utils.plot_features(singletonFeatureData,labels,utils._singleton_features_labels,args.opath)


def run_inference(opath, plot_text, modelName, figpath, model, particleDataTest, labelsTest, singletonDataTest, svTestingData=None, eventTestingData=None, pfMaskTestingData=None, svMaskTestingData=None):    
    model.load_state_dict(torch.load(args.mpath))
    eval_classifier(model, plot_text, modelName, figpath, particleDataTest, labelsTest, singletonDataTest,svTestingData=vertexDataTest,eventTestingData=eventTestingData,pfMaskTestingData=pfMaskTestingData,svMaskTestingData=svMaskTestingData,)
    return

def train_classifier(classifier, loss, batchSize, nepochs, modelName, outdir,
    #particleTrainingData, particleValidationData,  trainingLabels, validationLabels,
    #jetMassTrainingData=None, jetMassValidationData=None,
    #encoder=None,n_Dim=None, CorrDim=None, 
    #svTrainingData=None, svValidationData=None,
    #eventTrainingData=None, eventValidationData=None,
    #maskpfTrain=None, maskpfVal=None, masksvTrain=None, masksvVal=None, 
    ):

    optimizer = optim.Adam(classifier.parameters(), lr = 0.001)

    loss_vals_training = np.zeros(nepochs)
    loss_vals_validation = np.zeros(nepochs)
    acc_vals_training = np.zeros(nepochs)
    acc_vals_validation = np.zeros(nepochs)   

    is_pn = "PN" in model.name
    if is_pn:
        assert(maskpfTrain is not None)
    accuracy = Accuracy().to(device)
    model_dir = outdir
    os.system("mkdir -p ./"+model_dir)
    n_massbins = 20
       
    for iepoch in range(nepochs):
        loss_training, acc_training = [], []
        loss_validation, acc_validation = [], []
        print(f'Training Epoch {iepoch} on {len(train_loader.dataset)} jets')

        for istep, (x_pf, jet_features, jet_truthlabel) in enumerate(tqdm(train_loader)):

            if (args.test_run and istep>10 ): break
            x_pf = torch.nan_to_num(x_pf,nan=0.,posinf=0.,neginf=0.)
            #x_sv = torch.nan_to_num(x_pf,nan=0.,posinf=0.,neginf=0.)
            optimizer.zero_grad()
            #x_pf = x_pf.to(device)
            #jet_features = jet_features.to(device)
            #jet_truthlabel = jet_truthlabel.to(device)
            output = model(x_pf)
            #sys.exit(1)
            if args.loss == 'jsd':
                l = loss(output, jet_truthlabel, one_hots[istep*batchSize:(istep+1)*batchSize].to(device), n_massbins=n_massbins, LAMBDA_ADV=args.LAMBDA_ADV)
            elif 'disco' in args.loss:
                l = loss(output, jet_truthlabel, mass, LAMBDA_ADV=args.LAMBDA_ADV,)
            else:
                l = loss(output, jet_truthlabel)

            loss_training.append(l.item())
            acc_training.append(accuracy(output,torch.argmax(jet_truthlabel.squeeze(), dim=1)).cpu().detach().numpy())

            l.backward()
            optimizer.step()

            torch.cuda.empty_cache()
            #del , output
        print(f'Validating Epoch {iepoch} on {len(val_loader.dataset)} jets')
        for istep, (x_pf, jet_features, jet_truthlabel) in enumerate(tqdm(val_loader)):
            if (args.test_run and istep>10 ): break
            x_pf = torch.nan_to_num(x_pf,nan=0.,posinf=0.,neginf=0.)
            #x_sv = torch.nan_to_num(x_pf,nan=0.,posinf=0.,neginf=0.)

            #x_pf = x_pf.to(device)
            #jet_features = jet_features.to(device)
            #jet_truthlabel = jet_truthlabel.to(device)
            

            l_val = loss(output, jet_truthlabel)
            loss_validation.append(l_val.item())
            acc_validation.append(accuracy(output,torch.argmax(jet_truthlabel.squeeze(), dim=1)).cpu().detach().numpy())

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
                    svTestingData=None,eventTestingData=None,encoder=None,pfMaskTestingData=None,svMaskTestingData=None):
  
    #classifier.eval()  
    with torch.no_grad():
        print("Running predictions on test data")
        predictions = []

        batch_size = 1000
 
        ##### ParticleNet
        if "PN" in classifier.name:
            for istep, (subtensor,subtensorSV,subtensorpfMask,subtensorsvMask,subtensorE) in enumerate(zip(np.array_split(particleTestingData,batch_size),np.array_split(svTestingData,batch_size),np.array_split(pfMaskTestingData,batch_size),np.array_split(svMaskTestingData,batch_size),np.array_split(eventTestingData,batch_size))):
            #for istep in tqdm(range(int(len(particleTestingData)/batch_size))): 
                #testInputs = torch.FloatTensor(particleTestingData[istep*batch_size:(istep+1)*batch_size]).to(device)
                #testInputsSV = torch.FloatTensor(svTestingData[istep*batch_size:(istep+1)*batch_size]).to(device)
                #testpfMask = torch.FloatTensor(pfMaskTestingData[istep*batch_size:(istep+1)*batch_size]).to(device)
                #testsvMask = torch.FloatTensor(svMaskTestingData[istep*batch_size:(istep+1)*batch_size]).to(device)
                testpfPoints = subtensor[:,1:3,:]
                testsvPoints = subtensorSV[:,-2:,:]
      
                #if eventTestingData is not None:
                #    testInputsE = torch.FloatTensor(eventTestingData[istep*batch_size:(istep+1)*batch_size]).to(device)
                
                #else:
                #    testInputsE=None
                #print("here",testInputsE.shape)
                #predictions.append(classifier(testpfPoints,testInputs,testpfMask,testsvPoints,testInputsSV,testsvMask,testInputsE).cpu().detach().numpy())    
                testpfPoints = torch.FloatTensor(testpfPoints).to(device)
                subtensor = torch.FloatTensor(subtensor).to(device)
                subtensorpfMask = torch.FloatTensor(subtensorpfMask).to(device)
                testsvPoints = torch.FloatTensor(testsvPoints).to(device)
                subtensorSV = torch.FloatTensor(subtensorSV).to(device)
                subtensorsvMask = torch.FloatTensor(subtensorsvMask).to(device)
                subtensorE = torch.FloatTensor(subtensorE).to(device)
                predictions.append(classifier(testpfPoints,subtensor,subtensorpfMask,testsvPoints,subtensorSV,subtensorsvMask,subtensorE).cpu().detach().numpy())
                if (args.test_run): break
        ##### IN PF+SV+event
        elif svTestingData is not None and eventTestingData is not None:
            for subtensor,subtensorSV,subtensorE in zip(np.array_split(particleTestingData,1000),np.array_split(svTestingData,1000),np.array_split(eventTestingData,1000)):

                testInputs = torch.FloatTensor(subtensor).to(device)
                testInputsSV = torch.FloatTensor(subtensorSV).to(device)
                testInputsE = torch.FloatTensor(subtensorE).to(device)
                predictions.append(classifier(testInputs,testInputsSV,testInputsE).cpu().detach().numpy())
                del testInputs, testInputsSV, testInputsE
                #if (args.test_run): break 
 
        ##### PF+SV
        elif svTestingData is not None:
            for subtensor,subtensorSV in zip(np.array_split(particleTestingData,1000),np.array_split(svTestingData,1000)):
             
                testInputs = torch.FloatTensor(subtensor).to(device)
                testInputsSV = torch.FloatTensor(subtensorSV).to(device)
                predictions.append(classifier(testInputs,testInputsSV).cpu().detach().numpy())
                del testInputs, testInputsSV
                #if (args.test_run): break 
        ##### PF 
        else:
            for subtensor in np.array_split(particleTestingData,1000):
                testInputs = torch.FloatTensor(subtensor).to(device)
                predictions.append(classifier(testInputs).cpu().detach().numpy())
                del testInputs
                #if (args.test_run): break 
    predictions = [item for sublist in predictions for item in sublist]
    predictions = np.array(predictions)#.astype(np.float32)
    os.system("mkdir -p "+outdir)
    np.save(outdir+"/predictions.npy", predictions)
    if 'all_vs_QCD' in args.loss:
        qcd_idxs = np.where(testingLabels.sum(axis=1)==0,True,False)
    else:
        qcd_idxs = testingLabels[:,-1].astype(bool)
        utils.plot_correlation(predictions[qcd_idxs,-1],testingSingletons[qcd_idxs,0], "QCD output score","QCD jet $m_{SD}$ (GeV)", np.linspace(0,1,100),np.linspace(40,350,40),outdir, "qcd_vs_mass")
        utils.sculpting_curves(predictions[qcd_idxs,-1], testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="QCD", inverted=False)

    utils.plot_roc_curve(testingLabels, predictions, training_text, outdir, modelName, all_vs_QCD="all_vs_QCD" in args.loss, QCD_only=True)

    if args.is_binary:
        
        prob_2prong = predictions[qcd_idxs,0]
        utils.sculpting_curves(prob_2prong, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="Z\'",inverted=True)

    else:
        utils.plot_correlation(predictions[qcd_idxs,0],testingSingletons[qcd_idxs,0], "bb vs QCD output score","QCD jet $m_{SD}$ (GeV)", np.linspace(0,1,100),np.linspace(40,350,40),outdir, "bb_vs_mass")
        utils.plot_correlation(predictions[qcd_idxs,1],testingSingletons[qcd_idxs,0], "cc vs QCD output score","QCD jet $m_{SD}$ (GeV)", np.linspace(0,1,100),np.linspace(40,350,40),outdir, "cc_vs_mass")
        utils.plot_correlation(predictions[qcd_idxs,2],testingSingletons[qcd_idxs,0], "qq vs QCD output score","QCD jet $m_{SD}$ (GeV)", np.linspace(0,1,100),np.linspace(40,350,40),outdir, "qq_vs_mass")
        prob_bb = predictions[qcd_idxs,0]
        utils.sculpting_curves(prob_bb, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="bb",inverted=True)
        prob_cc = predictions[qcd_idxs,1]
        utils.sculpting_curves(prob_cc, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="cc",inverted=True)
        prob_qq = predictions[qcd_idxs,2]
        utils.sculpting_curves(prob_qq, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="qq",inverted=True)

if args.make_PN:
    predictions = singletonData[:,[utils._singleton_labels.index("zpr_fj_particleNetMD_Xbb"), utils._singleton_labels.index("zpr_fj_particleNetMD_Xcc"), utils._singleton_labels.index("zpr_fj_particleNetMD_Xqq"), utils._singleton_labels.index("zpr_fj_particleNetMD_QCD")]]
    utils.plot_roc_curve(labels, predictions, args.plot_text, args.opath, "particleNet-MD", all_vs_QCD=False,QCD_only=True)
    
    qcd_idxs = labels[:,-1].astype(bool)
    prob_bb = predictions[qcd_idxs,0]
    utils.sculpting_curves(prob_bb, singletonData[qcd_idxs,:],"ParticleNet-MD:bb score", args.opath, "particleNet-MD-bb",inverted=True)
    prob_cc = predictions[qcd_idxs,1]
    utils.sculpting_curves(prob_cc, singletonData[qcd_idxs,:], "ParticleNet-MD:cc score", args.opath, "particleNet-MD-cc",inverted=True)
    prob_qq = predictions[qcd_idxs,2]
    utils.sculpting_curves(prob_qq, singletonData[qcd_idxs,:], "ParticleNet-MD:qq score", args.opath, "particleNet-MD-qq",inverted=True)
    prob_QCD = predictions[qcd_idxs,3]
    utils.sculpting_curves(prob_QCD, singletonData[qcd_idxs,:], "ParticleNet-MD:QCD score", args.opath, "particleNet-MD-QCD",inverted=False)
    labels = np.concatenate((np.expand_dims(np.sum(labels[:,:-1],axis=1),-1),np.expand_dims(labels[:,-1],-1)),axis=1)
    predictions = np.concatenate((np.expand_dims(np.sum(predictions[:,:-1],axis=1),-1),np.expand_dims(predictions[:,-1],-1)),axis=1)
    utils.plot_roc_curve(labels, predictions, args.plot_text, args.opath, "particleNet-MD-2prong", all_vs_QCD=False,)

    sys.exit(1)

if args.make_N2:
    predictions = singletonData[:,[utils._singleton_labels.index("zpr_fj_n2b1")]]
    predictions = (predictions - np.min(predictions)) / ( np.max(predictions) - np.min(predictions))
    print(predictions.shape) 
    predictions = np.concatenate((1-predictions,predictions),axis=1)
    print(predictions.shape) 
    utils.plot_roc_curve(labels, predictions, args.plot_text, args.opath, "N2", all_vs_QCD=False,)
    qcd_idxs = labels[:,-1].astype(bool)
    prob_QCD = predictions[qcd_idxs,0]
    utils.sculpting_curves(prob_QCD, singletonData[qcd_idxs,:],"N2:QCD score", args.opath, "N2",inverted=True)
    prob_N2 = predictions[qcd_idxs,1]
    utils.sculpting_curves(prob_N2, singletonData[qcd_idxs,:],"N2:2prong score", args.opath, "N2",inverted=False)
    
    sys.exit(1)
maskpfTrain = None
maskpfVal = None
maskpfTest = None
masksvTrain = None
masksvVal = None
masksvTest = None

if args.model =='DNN':
    model = models.DNN("DNN",particleDataTrain.shape[1]*particleDataTrain.shape[2],labelsTrain.shape[1]).to(device)

elif args.model=='PN': 

    #model = models.ParticleNetTagger("PN",n_particles,n_vertex,labelsTrain.shape[1])
    if args.PN_v1:
        model = models.ParticleNetTagger("PN",particleDataTrain.shape[2],vertexDataTrain.shape[2],labelsTrain.shape[1],for_inference=_softmax, fc_params=[(128,0.1),(64,0.1)],conv_params=[(16, (64, 64, 64)),(16, (128, 128, 128)),],event_branch=args.event,sigmoid=_sigmoid)
    else: 
        model = models.ParticleNetTagger("PN",particleDataTrain.shape[2],vertexDataTrain.shape[2],labelsTrain.shape[1],for_inference=_softmax,event_branch=args.event,sigmoid=_sigmoid)
    # Batch,Nparts,Nfeatures
    #maskpfTrain = np.zeros((particleDataTrain.shape[0],particleDataTrain.shape[1]),dtype=bool)
    maskpfTrain = np.where(particleDataTrain[:,:,0]>0.,1., 0.)
    maskpfVal = np.where(particleDataVal[:,:,0]>0., 1., 0.)
    maskpfTest = np.where(particleDataTest[:,:,0]>0., 1., 0.)
    masksvTrain = np.where(vertexDataTrain[:,:,0]>0., 1., 0.)
    masksvVal = np.where(vertexDataVal[:,:,0]>0., 1., 0.)
    masksvTest = np.where(vertexDataTest[:,:,0]>0., 1., 0.)

    print("mask shape",maskpfTrain.shape)
    
    #maskpfTrain = np.repeat(maskpfTrain,6, axis=2) 
    maskpfTrain = np.expand_dims(maskpfTrain,axis=-1)
    maskpfVal = np.expand_dims(maskpfVal,axis=-1)
    maskpfTest = np.expand_dims(maskpfTest,axis=-1)
    masksvTrain = np.expand_dims(masksvTrain,axis=-1)
    masksvVal = np.expand_dims(masksvVal,axis=-1)
    masksvTest = np.expand_dims(masksvTest,axis=-1)

    maskpfTrain = np.swapaxes(maskpfTrain,1,2)
    maskpfVal = np.swapaxes(maskpfVal,1,2)
    maskpfTest = np.swapaxes(maskpfTest,1,2)
    masksvTrain = np.swapaxes(masksvTrain,1,2)
    masksvVal = np.swapaxes(masksvVal,1,2)
    masksvTest = np.swapaxes(masksvTest,1,2)

    particleDataTrain = np.swapaxes(particleDataTrain,1,2)
    particleDataVal = np.swapaxes(particleDataVal,1,2)
    particleDataTest = np.swapaxes(particleDataTest,1,2)
    vertexDataTrain = np.swapaxes(vertexDataTrain,1,2)
    vertexDataVal = np.swapaxes(vertexDataVal,1,2)
    vertexDataTest = np.swapaxes(vertexDataTest,1,2)
    print("particle shape",particleDataTrain.shape)
    #sys.exit(1)

elif args.model=='IN_SV_event':
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

elif args.model=='IN_SV': 
    model = models.GraphNetv2("IN_SV",n_particles,labelsTrain.shape[1],6,n_vertices=5, params_v=13, pv_branch=True, hidden=args.hidden, De=args.De, Do=args.Do,sigmoid=_sigmoid,softmax=_softmax)

    #somehow the (parts,features) axes get flipped in the IN 
    particleDataTrain = np.swapaxes(particleDataTrain,1,2)
    particleDataVal = np.swapaxes(particleDataVal,1,2)
    particleDataTest = np.swapaxes(particleDataTest,1,2)
    vertexDataTrain = np.swapaxes(vertexDataTrain,1,2)
    vertexDataVal = np.swapaxes(vertexDataVal,1,2)
    vertexDataTest = np.swapaxes(vertexDataTest,1,2)

elif args.model=='IN_noSV': 
    model = models.GraphNetv2("IN_noSV",n_particles,labelsTrain.shape[1],6,hidden=args.hidden, De=args.De, Do=args.Do,sigmoid=_sigmoid,softmax=_softmax,event_branch=False,)

    #somehow the (parts,features) axes get flipped in the IN 
    particleDataTrain = np.swapaxes(particleDataTrain,1,2)
    particleDataVal = np.swapaxes(particleDataVal,1,2)
    particleDataTest = np.swapaxes(particleDataTest,1,2)

elif args.model=='transformer':

    model = models.Transformer(args,"transformer",_softmax)
else:
    raise ValueError("Don't understand model ", args.model) 

model = model.to(device)
outdir = f"./{args.opath}/{model.name.replace(' ','_')}"
outdir = utils.makedir(outdir)
 
if args.mpath:
    run_inference(args.mpath, args.plot_text, model.name, args.mpath+"_plots", 
                  model, particleDataTest, labelsTest, singletonDataTest, svTestingData=vertexDataTest, eventTestingData=singletonFeatureDataTest,
                  pfMaskTestingData=maskpfTest,svMaskTestingData=masksvTest,
    )

else: 
    model = train_classifier(model, loss, batchsize, n_epochs, model.name, outdir+"/models/", 
                             #particleDataTrain, particleDataVal, labelsTrain, labelsVal, jetMassTrainingData=singletonDataTrain[:,0],
                             #jetMassValidationData=singletonDataVal[:,0],
                             #svTrainingData=vertexDataTrain, svValidationData=vertexDataVal,
                             #eventTrainingData=singletonFeatureDataTrain, eventValidationData=singletonFeatureDataVal,
                             #maskpfTrain=maskpfTrain, maskpfVal=maskpfVal, masksvTrain=masksvTrain, masksvVal=masksvVal,
    )
    eval_classifier(model, args.plot_text, model.name, outdir+"/plots/", particleDataTest, labelsTest, singletonDataTest, svTestingData=vertexDataTest, eventTestingData=singletonFeatureDataTest,pfMaskTestingData=maskpfTest,svMaskTestingData=masksvTest,) 

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




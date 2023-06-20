import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import numpy as np
import h5py
import json

import sklearn
import numpy.random as random
import corner
import builtins
import scipy
import time
from tqdm import tqdm 
import utils 
import sys
import glob
import models
from losses import *

# Imports neural net tools
import itertools
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd.variable import *
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score,  auc
from torchmetrics import Accuracy
from torchsummary import summary
from sklearn.decomposition import PCA
import torchsummary
from sklearn.preprocessing import OneHotEncoder
from loguru import logger

# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.sigmoid(out)
#         return out

class PreTrainer:
    def __init__(
        self,
        model,
        train_data,
        val_data,
        optimizer,
        save_every,
        outdir, 
        loss,
        max_epochs,
        args
        
    ):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.args = args
        self.outdir = outdir
        self.loss = loss
        self.loss_vals_training = np.zeros(max_epochs)
        self.loss_vals_validation = np.zeros(max_epochs)
       
        self.name = self.model.name
        self.model = DDP(self.model, device_ids=[self.gpu_id],find_unused_parameters=True)
        #self.MLP = MLP(args.n_out_nodes,16,1).to(self.gpu_id)
        #self.MLP_optimizer = optim.Adam(self.MLP.parameters(), lr = args.lr)
        


    def _run_epoch_val(self, epoch):
        b_sz = len(next(iter(self.val_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.val_data)}")
        loss_validation, acc_validation = [], []
        self.model.train(False)
        for istep, (x_pf, x_sv, jet_features, label) in enumerate(tqdm(self.val_data)):
                lv = self._run_batch_val(istep,x_pf, x_sv, jet_features, label)
                loss_validation.append(lv)
              
        epoch_val_loss = np.mean(loss_validation)
       
        self.loss_vals_validation[epoch] = epoch_val_loss
        
        
    def _run_batch_val(self, istep, x_pf, x_sv, jet_features, label):
        
        
        

        self.model.train(False)
        accuracy = Accuracy().to(self.gpu_id)

        x_pf = torch.nan_to_num(x_pf,nan=0.,posinf=0.,neginf=0.)
        x_sv = torch.nan_to_num(x_sv,nan=0.,posinf=0.,neginf=0.)
        
        for param in self.model.parameters():
            param.grad = None
        self.optimizer.zero_grad()
        if not self.args.load_gpu:
            x_pf = x_pf.to(self.gpu_id)
            x_sv = x_sv.to(self.gpu_id)
            
            
            jet_features = jet_features.to(self.gpu_id)
            
            label = label.to(self.gpu_id)
        if self.args.sv:
            output = self.model(x_pf,x_sv)
        else:
            output = self.model(x_pf)
            
        loss_fn = margin_triplet_loss
        
            
        l = loss_fn(output, label)

        return l.item()
        
        
    def _run_batch_train(self, istep, x_pf, x_sv, jet_features, label):
       

        #Unpacking pairs
       
        
        
        
               
        
        self.model.train(True)
        accuracy = Accuracy().to(self.gpu_id)
#         if 'all_vs_QCD' in self.args.loss:
#             label = label[:,:-]

        #if (self.args.test_run and istep>10 ): break
        x_pf = torch.nan_to_num(x_pf,nan=0.,posinf=0.,neginf=0.)
        x_sv = torch.nan_to_num(x_sv,nan=0.,posinf=0.,neginf=0.)
        
        
        for param in self.model.parameters():
            param.grad = None
        
        if not self.args.load_gpu:
            x_pf = x_pf.to(self.gpu_id)
            x_sv = x_sv.to(self.gpu_id)
        
            
            jet_features = jet_features.to(self.gpu_id)
            label = label.to(self.gpu_id)
        if self.args.sv:
            output = self.model(x_pf,x_sv)
           
        else:
            output = self.model(x_pf)
            
        loss_fn = margin_triplet_loss
        
        self.optimizer.zero_grad()
#         self.MLP_optimizer.zero_grad()
        
        
        
            
        l = loss_fn(output, label)
        l.backward()
#         l_p.backward()
        
        self.optimizer.step()
        

        torch.cuda.empty_cache()
        return l.item()

    def _run_epoch_train(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        loss_training = []
        loss_validation = []
        self.model.train(True)
        for istep, (x_pf, x_sv, jet_features, label) in enumerate(tqdm(self.train_data)):
                lt = self._run_batch_train(istep,x_pf, x_sv, jet_features, label)
                
                loss_training.append(lt)
                
                
        
        epoch_train_loss = np.mean(loss_training)
        
        self.loss_vals_training[epoch] = epoch_train_loss
        
    def _save_snapshot(self, epoch):
        torch.save(self.model.state_dict(),"{}/epoch_{}_{}_loss_{}_{}.pth".format(self.outdir,epoch,self.name.replace(' ','_'),round(self.loss_vals_training[epoch],4),round(self.loss_vals_validation[epoch],4)))
        print(f" Training snapshot saved")

    def train(self, max_epochs: int):
        self.model.train(True)
        np.random.seed(max_epochs)
        random.seed(max_epochs)
        
    
#         if self.args.distributed: 
#             train_loader.sampler.set_epoch(nepochs)
          
    
        
        model_dir = self.outdir
        os.system("mkdir -p ./"+model_dir)
        n_massbins = 20
    
        if self.args.continue_training:
            self.model.load_state_dict(torch.load(self.args.mpath))
            start_epoch = self.args.mpath.split("/")[-1].split("epoch_")[-1].split("_")[0]
            start_epoch = int(start_epoch) + 1
            print(f"Continuing training from epoch {start_epoch}...")
        else:
            start_epoch = 1
        end_epoch = max_epochs
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch_train(epoch)
            self._run_epoch_val(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        self.run_inference(self.args.plot_text,self.val_data,self.args)
        torch.cuda.empty_cache()
        utils.plot_loss(self.loss_vals_training,self.loss_vals_validation,model_dir)
    
    def run_inference(self, training_text,val_loader,args):
        with torch.no_grad():
            print("Running predictions on test data")
            cspace = []
            testingLabels = []
            
            batch_size = 1000
    
            for istep, (x_pf, x_sv, jet_features, label) in enumerate(tqdm(val_loader)):
                #model.eval()
                label = torch.sum(label[:,0:3],dim=1)
                x_pf = x_pf.to(self.gpu_id)
                x_sv = x_sv.to(self.gpu_id)
                label = label.to(self.gpu_id)
                testingLabels.append(label.cpu().detach().numpy())
                if args.sv:
                    cspace.append(self.model(x_pf,x_sv).cpu().detach().numpy())
                else: 
                    cspace.append(self.model(x_pf).cpu().detach().numpy())
                torch.cuda.empty_cache()
                #break
            
        self.outdir = self.outdir + '/plots'
        cspace = np.vstack(cspace)#.astype(np.float32)
        testingLabels = np.concatenate(testingLabels,axis=0)
#         cpspace = np.array(cspace)
#         testingLabels = np.array(testingLabels)
        
    
        os.system("mkdir -p "+self.outdir)
        N = len(cspace)  # Number of instances
        D = len(cspace[0]) # Number of dimensions
        labels = testingLabels  # Binary labels
        
        data = cspace

        # Apply PCA to reduce dimensionality to 2 for visualization
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)

        # Separate data points based on labels
        data_label0 = data_pca[labels == 0.]
        data_label1 = data_pca[labels == 1.]

        # Create the PCA plot
        plt.scatter(data_label0[:, 0], data_label0[:, 1], label='QCD')
        plt.scatter(data_label1[:, 0], data_label1[:, 1], label='Z_p')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Plot')
        plt.legend()
        plt.savefig(f'{self.outdir}/PCA_Constrastive_Spaces.png')
        plt.savefig(f'{self.outdir}/PCA_Constrastive_Spaces.pdf')
        
       
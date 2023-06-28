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
from PIL import Image

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
import matplotlib.lines as mlines
from sklearn.decomposition import PCA
import torchsummary
from sklearn.preprocessing import OneHotEncoder
from loguru import logger

import contrastive_losses

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
        loss_fn = contrastive_losses.SimCLRLoss()

        output = torch.unsqueeze(output,dim=1)
        l = loss_fn.forward2(output, torch.argmax(label, dim=1))

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
        
        
        
        self.optimizer.zero_grad()
        
        
        
        
        
        loss_fn = contrastive_losses.SimCLRLoss()

        output = torch.unsqueeze(output,dim=1)
        l = loss_fn.forward2(output, torch.argmax(label, dim=1))
        
        l.backward()
        
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
    

    


    def overlaid_corner(self,samples_list, sample_labels):
        CORNER_KWARGS = dict(
        smooth=0.9,
        label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16),
        quantiles=[0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        show_titles=True,
        max_n_ticks=3,
    )
        """Plots multiple corners on top of each other"""
        # get some constants
        n = len(samples_list)
        _, ndim = samples_list[0].shape
        max_len = max([len(s) for s in samples_list])
        cmap = plt.cm.get_cmap('gist_rainbow', n)
        colors = [cmap(i) for i in range(n)]

        plot_range = []
        for dim in range(ndim):
            plot_range.append(
                [
                    min([min(samples_list[i].T[dim]) for i in range(n)]),
                    max([max(samples_list[i].T[dim]) for i in range(n)]),
                ]
            )

        CORNER_KWARGS.update(range=plot_range)

        fig = corner.corner(
            samples_list[0],
            color=colors[0],
            **CORNER_KWARGS
        )

        for idx in range(1, n):
            fig = corner.corner(
                samples_list[idx],
                fig=fig,
                weights=self.get_normalisation_weight(len(samples_list[idx]), max_len),
                color=colors[idx],
                **CORNER_KWARGS
            )

        plt.legend(
            handles=[
                mlines.Line2D([], [], color=colors[i], label=sample_labels[i])
                for i in range(n)
            ],
            fontsize=20, frameon=False,
            bbox_to_anchor=(1, ndim), loc="upper right"
        )
        plt.savefig(f'{self.outdir}/Corner_Constrastive_Spaces.png')
        plt.savefig(f'{self.outdir}/Corner_Constrastive_Spaces.pdf')
        plt.close()


    def get_normalisation_weight(self,len_current_samples, len_of_longest_samples):
        return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)
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
            start_epoch = 0
        end_epoch = max_epochs
        if start_epoch < max_epochs:
            for epoch in range(self.epochs_run, max_epochs):
                epoch = epoch + start_epoch
                if epoch<max_epochs:
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
                label = torch.argmax(label,dim=1)
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
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)

        # Separate data points based on labels
        data_label0 = data_pca[labels == 0]
        data_label1 = data_pca[labels == 1]
        data_label2 = data_pca[labels == 2]
        data_label3 = data_pca[labels == 3]

        # Plot the data points
        plt.scatter(data_label0[:, 0], data_label0[:, 1], label='QCD',alpha = 0.05)
        plt.scatter(data_label1[:, 0], data_label1[:, 1], label='Z_1',alpha = 0.05)
        plt.scatter(data_label2[:, 0], data_label2[:, 1], label='Z_2',alpha = 0.05)
        plt.scatter(data_label3[:, 0], data_label3[:, 1], label='Z_3',alpha = 0.05)
        plt.legend()
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('2D PCA')
        filename = 'PCA2D'
        plt.savefig(os.path.join(self.outdir,filename))
        
        # Apply PCA to reduce dimensionality to 2 for visualization
        pca = PCA(n_components=3)
        data_pca = pca.fit_transform(data)

        # Separate data points based on labels
        data_label0 = data_pca[labels == 0.]
        data_label1 = data_pca[labels == 1.]
        data_label2 = data_pca[labels == 2.]
        data_label3 = data_pca[labels == 3.]

        # Create a list to store the image frames
        frames = []

        # Create 3D PCA plots at different angles
        for angle in range(0, 360, 5):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data_label0[:, 0], data_label0[:, 1], data_label0[:, 2], label='QCD',alpha = 0.05)
            ax.scatter(data_label1[:, 0], data_label1[:, 1], data_label1[:, 2], label='Z_1',alpha = 0.05)
            ax.scatter(data_label2[:, 0], data_label2[:, 1], data_label2[:, 2], label='Z_2',alpha = 0.05)
            ax.scatter(data_label3[:, 0], data_label3[:, 1], data_label3[:, 2], label='Z_3',alpha = 0.05)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title('3D PCA Plot')
            ax.legend()
            ax.grid(False)

            ax.view_init(elev=10, azim=angle)  # Set the elevation and azimuth angles

            # Save the plot as an image file
            filename = f'PCA_Constrastive_Spaces_{angle}.png'
            fig.savefig(os.path.join(self.outdir,filename))
            plt.close(fig)
        for angle in range(0, 360, 5):
            filename = f'PCA_Constrastive_Spaces_{angle}.png'
            # Open the saved image and append it to the list of frames
            img = Image.open(os.path.join(self.outdir,filename))
            frames.append(img)

        # Save the frames as a GIF
        gif_filename = os.path.join(self.outdir,'pca_animation.gif')
        frames[0].save(gif_filename, format='GIF', append_images=frames[1:], save_all=True, duration=200, loop=1)


        
        
        
        self.overlaid_corner((data[labels == 0.],data[labels == 1.],data[labels == 2.],data[labels == 3.]),('QCD','Z_1','Z_2','Z_3'))
        

       
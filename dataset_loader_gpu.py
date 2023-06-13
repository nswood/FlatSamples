import numpy as np
import h5py
import glob
import tqdm
import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class zpr_loader(Dataset):
    def __init__(self, raw_paths, qcd_only=True, transform=None,maxfiles=None):
        #super(zpr_loader, self).__init__(raw_paths)
        #self.strides = [0]
        #self.ratio = ratio
        self.raw_paths = sorted(glob.glob(raw_paths+'*h5'))[:maxfiles]
        self.fill_data()

        #self.calculate_offsets()
    def calculate_offsets(self):
        for path in self.raw_paths:
            print(path)
            with h5py.File(path, 'r') as f:
                self.strides.append(f['features'].shape[0])
        self.strides = np.cumsum(self.strides)

    def fill_data(self):
        print("Reading files...")
        self.data_features = []
        self.data_sv_features = [] 
        self.data_jetfeatures = []
        self.data_truthlabel = [] 

        for fi,path in enumerate(tqdm.tqdm(self.raw_paths)):
            with h5py.File(path, 'r') as f:

                 tmp_features = f['features'][()].astype(np.float32)
                 tmp_sv_features = f['features_SV'][()].astype(np.float32)
                 tmp_jetfeatures = f['jet_features'][()].astype(np.float32)
                 tmp_truthlabel = f['jet_truthlabel'][()]
                 self.data_features.append(tmp_features)
                 self.data_sv_features.append(tmp_sv_features)
                 self.data_jetfeatures.append(tmp_jetfeatures)
                 self.data_truthlabel.append(tmp_truthlabel)
                 #if fi == 0:
                 #    self.data_features = tmp_features
                 #    self.data_sv_features = tmp_sv_features
                 #    self.data_jetfeatures = tmp_jetfeatures
                 #    self.data_truthlabel = tmp_truthlabel
                 #else:
                 #    self.data_features = np.concatenate((self.data_features,tmp_features))
                 #    self.data_sv_features = np.concatenate((self.data_sv_features,tmp_sv_features))
                 #    self.data_jetfeatures = np.concatenate((self.data_jetfeatures,tmp_jetfeatures))
                 #    self.data_truthlabel = np.concatenate((self.data_truthlabel,tmp_truthlabel))

        self.data_features = [item for sublist in self.data_features for item in sublist]
        self.data_sv_features = [item for sublist in self.data_sv_features for item in sublist]
        self.data_jetfeatures = [item for sublist in self.data_jetfeatures for item in sublist]
        self.data_truthlabel = [item for sublist in self.data_truthlabel for item in sublist]

        self.data_features = np.array(self.data_features)
        self.data_sv_features = np.array(self.data_sv_features)
        self.data_jetfeatures = np.array(self.data_jetfeatures)
        self.data_truthlabel = np.array(self.data_truthlabel)

        print("self.data_features.shape",self.data_features.shape)
     
        self.data_features = torch.cuda.FloatTensor(self.data_features)
        self.data_sv_features = torch.cuda.FloatTensor(self.data_sv_features)
        self.data_jetfeatures = torch.cuda.FloatTensor(self.data_jetfeatures)
        self.data_truthlabel = torch.cuda.FloatTensor(self.data_truthlabel)
    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.h5')))
        return raw_files

    @property
    def processed_file_names(self):
        return []

    def __len__(self):
        return self.data_jetfeatures.shape[0]#self.strides[-1]

    def __getitem__(self, idx):
        x_pf = self.data_features[idx,:,:]
        x_sv = self.data_sv_features[idx,:,:]
        x_jet = self.data_jetfeatures[idx,:]
        y = self.data_truthlabel[idx]
        return x_pf, x_sv, x_jet, y

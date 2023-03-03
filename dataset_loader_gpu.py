import numpy as np
import h5py
import glob
import tqdm
import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class zpr_loader(Dataset):
    def __init__(self, raw_paths, qcd_only=True, transform=None):
        #super(zpr_loader, self).__init__(raw_paths)
        #self.strides = [0]
        #self.ratio = ratio
        self.raw_paths = glob.glob(raw_paths+'*_0_*h5')
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
        for fi,path in enumerate(tqdm.tqdm(self.raw_paths)):
            with h5py.File(path, 'r') as f:

                 tmp_features = f['features'][()].astype(np.float32)
                 tmp_jetfeatures = f['features'][()].astype(np.float32)
                 tmp_truthlabel = f['jet_truthlabel'][()]
                 if fi == 0:
                     self.data_features = tmp_features
                     self.data_jetfeatures = tmp_jetfeatures
                     self.data_truthlabel = tmp_truthlabel
                 else:
                     self.data_features = np.concatenate((self.data_features,tmp_features))
                     self.data_jetfeatures = np.concatenate((self.data_jetfeatures,tmp_jetfeatures))
                     self.data_truthlabel = np.concatenate((self.data_truthlabel,tmp_truthlabel))

                 if self.data_truthlabel.shape[0] > 10000: break
        self.data_features = torch.cuda.FloatTensor(self.data_features)
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
        x_jet = self.data_jetfeatures[idx,:]
        y = self.data_truthlabel[idx]
        return x_pf, x_jet, y

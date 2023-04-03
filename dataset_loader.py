import numpy as np
import h5py
import glob

import torch
from torch.utils.data.dataset import Dataset  # For custom datasets


class zpr_loader(Dataset):
    def __init__(self, raw_paths, qcd_only=True, transform=None, maxfiles=None):
        #super(zpr_loader, self).__init__(raw_paths)
        self.strides = [0]
        #self.ratio = ratio
        self.raw_paths = glob.glob(raw_paths+'*h5')[:maxfiles]
        self.calculate_offsets()
    def calculate_offsets(self):
        for path in self.raw_paths:
            print(path)
            with h5py.File(path, 'r') as f:
                self.strides.append(f['features'].shape[0])
        self.strides = np.cumsum(self.strides)

    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.h5')))
        return raw_files

    @property
    def processed_file_names(self):
        return []

    def __len__(self):
        return self.strides[-1]

    def __getitem__(self, idx):

        file_idx = np.searchsorted(self.strides, idx) - 1
        if file_idx < 0:
            file_idx = 0
        idx_in_file = idx - self.strides[max(0, file_idx)]  # - 1

        if idx in self.strides and idx_in_file != 0:
            idx_in_file = 0
            file_idx += 1

        if file_idx >= self.strides.size:
            raise Exception(f'{idx} is beyond the end of the event list {self.strides[-1]}')
        #edge_index = torch.empty((2,0), dtype=torch.long)
        #print(self.raw_paths[file_idx])
        with h5py.File(self.raw_paths[file_idx],'r') as f:

            #Npfs = np.count_nonzero(f['features'][idx_in_file,:,0])


            x_pf = torch.FloatTensor(np.array(f['features'][idx_in_file,:,:],dtype=np.float32))
            #x_sv = torch.cuda.FloatTensor(f['features_SV'][idx_in_file,:,:]).cuda()

            y = torch.FloatTensor(np.array(f['jet_truthlabel'][idx_in_file],dtype=np.float32))
            x_jet = torch.FloatTensor(np.array(f['jet_features'][idx_in_file],dtype=np.float32))

            return {"x_pf":x_pf, "x_jet": x_jet, "y":y}

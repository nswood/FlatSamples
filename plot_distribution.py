import numpy as np
import h5py
import keras.backend as K
import tensorflow as tf
import json

# Imports neural net tools                                                                                                                                                                                                                                    

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv1D, Lambda, Dot, Flatten
from keras.models import Model


print("Extracting")

fOne = h5py.File("data/FullQCD_FullSig_Zqq.h5", 'r')
totalData = fOne["deepDoubleQ"][:]
print(totalData.shape)
modelName = "IN_FlatSamples_fullQCDfullSig"

# Sets controllable values                                                                                                                                                                                                                                    

particlesConsidered = 50
entriesPerParticle = 4

eventDataLength = 6

decayTypeColumn = -1

testDataLength = int(len(totalData)*0.1)
trainingDataLength = int(len(totalData)*0.8)

validationDataLength = int(len(totalData)*0.1)
particleDataLength = particlesConsidered * entriesPerParticle

labels = totalData[:, decayTypeColumn:]
particleData = totalData[:, eventDataLength:particleDataLength + eventDataLength]

pt = particleData[:, :50]
eta = particleData[:, 50:100]
phi = particleData[:, 100:150]
charge = particleData[:, 150:200]

pt_sig = [particleData[i, :50] for i in range(len(particleData)) if labels[i] == 1]
pt_bkg = [particleData[i, :50] for i in range(len(particleData)) if labels[i] == 0]
eta_sig = [particleData[i, 50:100] for i in range(len(particleData)) if labels[i] == 1]
eta_bkg = [particleData[i, 50:100] for i in range(len(particleData)) if labels[i] == 0]
phi_sig = [particleData[i, 100:150] for i in range(len(particleData)) if labels[i] == 1]
phi_bkg = [particleData[i, 100:150] for i in range(len(particleData)) if labels[i] == 0]
charge_sig = [particleData[i, 150:200] for i in range(len(particleData)) if labels[i] == 1]
charge_bkg = [particleData[i, 150:200] for i in range(len(particleData)) if labels[i] == 0]

jet_eta_sig = [totalData[i, 0] for i in range(len(totalData)) if labels[i] == 1] 
jet_phi_sig = [totalData[i, 1] for i in range(len(totalData)) if labels[i] == 1]
jet_EhadOverEem_sig = [totalData[i, 2] for i in range(len(totalData)) if labels[i] == 1]
jet_sdmass_sig = [totalData[i, 3] for i in range(len(totalData)) if labels[i] == 1]
jet_pT_sig = [totalData[i, 4] for i in range(len(totalData)) if labels[i] == 1]
jet_mass_sig = [totalData[i, 5] for i in range(len(totalData)) if labels[i] == 1]

jet_eta_bkg = [totalData[i, 0] for i in range(len(totalData)) if labels[i] == 0]                                                                                                                                                                             
jet_phi_bkg = [totalData[i, 1] for i in range(len(totalData)) if labels[i] == 0]                                                                                                                                                                             
jet_EhadOverEem_bkg = [totalData[i, 2] for i in range(len(totalData)) if labels[i] == 0]
jet_sdmass_bkg = [totalData[i, 3] for i in range(len(totalData)) if labels[i] == 0]
jet_pT_bkg = [totalData[i, 4] for i in range(len(totalData)) if labels[i] == 0]
jet_mass_bkg = [totalData[i, 5] for i in range(len(totalData)) if labels[i] == 0]    

import matplotlib.pyplot as plt
plt.figure()
plt.title('pT sig')
plt.hist(np.array(pt_sig).flatten(), bins = 20)
plt.savefig('data/hist/pt_sig_hist.jpg')

plt.figure()
plt.title('pT bkg')
plt.hist(np.array(pt_bkg).flatten(), bins = 20)
plt.savefig('data/hist/pt_bkg_hist.jpg')

plt.figure()
plt.title('pT')
plt.hist(np.array(pt_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(pt_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right') 
plt.savefig('data/hist/pt_dual_hist.jpg')

plt.figure()
plt.title('eta sig')
plt.hist(np.array(eta_sig).flatten(), bins = 20)
plt.savefig('data/hist/eta_sig_hist.jpg')

plt.figure()
plt.title('eta_bkg')
plt.hist(np.array(eta_bkg).flatten(), bins = 20)
plt.savefig('data/hist/eta_bkg_hist.jpg')

plt.figure()
plt.title('eta')
plt.hist(np.array(eta_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(eta_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/eta_dual_hist.jpg')

plt.figure()
plt.title('phi sig')
plt.hist(np.array(phi_sig).flatten(), bins = 20)
plt.savefig('data/hist/phi_sig_hist.jpg')

plt.figure()
plt.title('phi bkg')
plt.hist(np.array(phi_bkg).flatten(), bins = 20)
plt.savefig('data/hist/phi_bkg_hist.jpg')

plt.figure()
plt.title('phi')
plt.hist(np.array(phi_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(phi_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/phi_dual_hist.jpg')

plt.figure()
plt.title('charge sig')
plt.hist(np.array(charge_sig).flatten(), bins = 20)
plt.savefig('data/hist/charge_sig_hist.jpg')

plt.figure()
plt.title('charge bkg')
plt.hist(np.array(charge_bkg).flatten(), bins = 20)
plt.savefig('data/hist/charge_bkg_hist.jpg')

plt.figure()
plt.title('charge')
plt.hist(np.array(charge_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(charge_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/charge_dual_hist.jpg')


plt.figure()
plt.title('jet_eta')
plt.hist(np.array(jet_eta_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(jet_eta_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/jet_eta_dual_hist.jpg')

plt.figure()
plt.title('jet_mass')
plt.hist(np.array(jet_mass_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(jet_mass_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/jet_mass_dual_hist.jpg')

plt.figure()
plt.title('jet_phi')
plt.hist(np.array(jet_phi_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(jet_phi_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/jet_phi_dual_hist.jpg')

plt.figure()
plt.title('jet_sdmass')
plt.hist(np.array(jet_sdmass_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(jet_sdmass_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/jet_sdmass_dual_hist.jpg')

plt.figure()
plt.title('jet_pT')
plt.hist(np.array(jet_pT_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(jet_pT_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/jet_pT_dual_hist.jpg')


plt.figure()
plt.title('pT, mass signal histogram')
plt.hist2d(np.array(jet_pT_sig).flatten(), np.array(jet_mass_sig).flatten())
plt.colorbar()
plt.xlabel('jet_pT')
plt.ylabel('jet_mass')
plt.savefig('data/hist/jet_pT_mass_hist2d_sig.jpg')


plt.figure()
plt.title('pT, mass bkg histogram')
plt.hist2d(np.array(jet_pT_bkg).flatten(), np.array(jet_mass_bkg).flatten())
plt.colorbar()
plt.xlabel('jet_pT')
plt.ylabel('jet_mass')
plt.savefig('data/hist/jet_pT_mass_hist2d_bkg.jpg')
 



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
print(totalData[:, 0])
print(totalData[:, 1])
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
np.random.shuffle(totalData)

labels = totalData[:, decayTypeColumn:]
particleData = totalData[:, eventDataLength:particleDataLength + eventDataLength]
testData = totalData[trainingDataLength + validationDataLength:, ]
particleTestData = np.transpose(particleData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerParticle, particlesConsidered),
                                axes=(0, 2, 1))
testLabels = np.array(labels[trainingDataLength + validationDataLength:])


# Defines the interaction matrices

# Defines the recieving matrix for particles
RR = []
for i in range(particlesConsidered):
    row = []
    for j in range(particlesConsidered * (particlesConsidered - 1)):
        if j in range(i * (particlesConsidered - 1), (i + 1) * (particlesConsidered - 1)):
            row.append(1.0)
        else:
            row.append(0.0)
    RR.append(row)
RR = np.array(RR)
RR = np.float32(RR)
RRT = np.transpose(RR)

# Defines the sending matrix for particles
RST = []
for i in range(particlesConsidered):
    for j in range(particlesConsidered):
        row = []
        for k in range(particlesConsidered):
            if k == j:
                row.append(1.0)
            else:
                row.append(0.0)
        RST.append(row)
rowsToRemove = []
for i in range(particlesConsidered):
    rowsToRemove.append(i * (particlesConsidered + 1))
RST = np.array(RST)
RST = np.float32(RST)
RST = np.delete(RST, rowsToRemove, 0)
RS = np.transpose(RST)


# Creates and trains the neural net

# Particle data interaction NN
inputParticle = Input(shape=(particlesConsidered, entriesPerParticle), name="inputParticle")

XdotRR = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RR, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRR")(inputParticle)
XdotRS = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RS, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRS")(inputParticle)
Bpp = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bpp")([XdotRR, XdotRS])

convOneParticle = Conv1D(80, kernel_size=1, activation="relu", name="convOneParticle")(Bpp)
convTwoParticle = Conv1D(50, kernel_size=1, activation="relu", name="convTwoParticle")(convOneParticle)
convThreeParticle = Conv1D(30, kernel_size=1, activation="relu", name="convThreeParticle")(convTwoParticle)

Epp = BatchNormalization(momentum=0.6, name="Epp")(convThreeParticle)

# Combined prediction NN
EppBar = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RRT, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="EppBar")(Epp)
C = Lambda(lambda listOfTensors: tf.concat((listOfTensors[0], listOfTensors[1]), axis=2), name="C")(
    [inputParticle, EppBar])

convPredictOne = Conv1D(80, kernel_size=1, activation="relu", name="convPredictOne")(C)
convPredictTwo = Conv1D(50, kernel_size=1, activation="relu", name="convPredictTwo")(convPredictOne)

O = Conv1D(24, kernel_size=1, activation="relu", name="O")(convPredictTwo)

# Calculate output
OBar = Lambda(lambda tensor: K.sum(tensor, axis=1), name="OBar")(O)

denseEndOne = Dense(60, activation="relu", name="denseEndOne")(OBar)
normEndOne = BatchNormalization(momentum=0.6, name="normEndOne")(denseEndOne)
denseEndTwo = Dense(30, activation="relu", name="denseEndTwo")(normEndOne)
denseEndThree = Dense(10, activation="relu", name="denseEndThree")(denseEndTwo)
output = Dense(1, activation="sigmoid", name="output")(denseEndThree)

print("Compiling")

model = Model(inputs=[inputParticle], outputs=[output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])


model.load_weights("./data/"+modelName+".h5")
print("Predicting")

predictions = model.predict([particleTestData])
print(predictions)
print(testLabels)
predictions = [[i[0], 1-i[0]] for i in predictions]
testLabels = np.array([[i[0], 1-i[0]] for i in testLabels])
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc, accuracy_score


fpr, tpr, threshold = roc_curve(np.array(testLabels).reshape(-1), np.array(predictions).reshape(-1))
plt.plot(fpr, tpr, lw=2.5, label="{}, AUC = {:.1f}\%".format('ZprimeAtoqq IN',auc(fpr,tpr)*100))
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.savefig('./data/'+modelName+'_ROC.jpg')


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

fpr, tpr, threshold = roc_curve(np.array(testLabels).reshape(-1), np.array(predictions).reshape(-1))
cuts = {}
for wp in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]: # % mistag rate
    idx, val = find_nearest(fpr, wp)
    cuts[str(wp)] = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate
        
f, ax = plt.subplots(figsize=(10,10))

print(testData[:,0])
print(testData[:,1])

sculpt_vars = ['jet_eta', "jet_phi","jet_EhadOverEem","jet_sdmass", 'jet_pT', 'jet_mass']
for i in range(len(sculpt_vars)):
    f, ax = plt.subplots(figsize=(10,10))

    for wp, cut in reversed(sorted(cuts.items())):
        predictions = np.array(predictions)
        ctdf = np.array([testData[pred, i] for pred in range(len(predictions)) if predictions[pred,0] > cut])
        weight = np.array([testLabels[pred, 1] for pred in range(len(predictions)) if predictions[pred,0] > cut])
        
        if str(wp)=='1.0':
            ax.hist(ctdf.flatten(), weights = weight/np.sum(weight), lw=2,
                        histtype='step',label='No tagging applied')
        else:
            ax.hist(ctdf.flatten(), weights = weight/np.sum(weight), lw=2,
                        histtype='step',label='{}%  mistagging rate'.format(float(wp)*100.))

    ax.set_xlabel(sculpt_vars[i])
    ax.set_ylabel('Normalized Scale')
    ax.set_title('Sculpting ' + sculpt_vars[i]) 
    ax.legend() 

    f.savefig('data/hist/sculpting_' + sculpt_vars[i] + '.jpg')


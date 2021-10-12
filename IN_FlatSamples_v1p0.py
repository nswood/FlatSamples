# Imports basics

import numpy as np
import h5py
import keras.backend as K
import tensorflow as tf
import json
import setGPU

# Imports neural net tools

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv1D, Lambda, Dot, Flatten
from keras.models import Model

# Opens files and reads data

print("Extracting")

fOne = h5py.File("data/FullQCD_FullSig_Zqq.h5", 'r')
totalData = fOne["deepDoubleQ"][:]
print(totalData.shape)

# Sets controllable values

particlesConsidered = 50
entriesPerParticle = 4

eventDataLength = 6

decayTypeColumn = -1

trainingDataLength = int(len(totalData)*0.8)

validationDataLength = int(len(totalData)*0.1)

numberOfEpochs = 100
batchSize = 1024

modelName = "IN_FlatSamples_fullQCDfullSig"

# Creates Training Data

print("Preparing Data")

particleDataLength = particlesConsidered * entriesPerParticle

np.random.shuffle(totalData)

labels = totalData[:, decayTypeColumn:]

particleData = totalData[:, eventDataLength:particleDataLength + eventDataLength]

particleTrainingData = np.transpose(
    particleData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerParticle, particlesConsidered),
    axes=(0, 2, 1))
trainingLabels = np.array(labels[0:trainingDataLength])

particleValidationData = np.transpose(
    particleData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength,
                                                                                         entriesPerParticle,
                                                                                         particlesConsidered),
    axes=(0, 2, 1))
validationLabels = np.array(labels[trainingDataLength:trainingDataLength + validationDataLength])

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

print('Calculating')

modelCallbacks = [EarlyStopping(patience=10),
                  ModelCheckpoint(filepath="./data/"+modelName+".h5", save_weights_only=True,
                                  save_best_only=True)]

history = model.fit([particleTrainingData], trainingLabels, epochs=numberOfEpochs, batch_size=batchSize,
                    callbacks=modelCallbacks,
                    validation_data=([particleValidationData], validationLabels))

with open("./data/"+modelName+",history.json", "w") as f:
    json.dump(history.history,f)

print("Loading weights")

model.load_weights("./data/"+modelName+".h5")

model.save("./data/"+modelName+",model")

print("Predicting")

predictions = model.predict([particleTestData])

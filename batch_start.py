from classes.Model import Model
from classes.MemoryLevel import MemoryLevel
from classes.Embedder import Embedder
from classes.FOM4 import FOM4
from classes.FOM3 import FOM3
from math import floor

import numpy as np
import scipy.io.wavfile as siow
import copy
#import tensorflow as tf
#from tensorflow.python.compiler.tensorrt import trt_convert as trt
# FOR REPRODUCIBILITY

import tensorflow as tf
import random 



def softReset(model):
    for l in range(1, memoryLevels):
        # Reset classes
        model.embedSystem[l].vd.reset()
        # Reset non-data levels
        model.memory.resetMemoryLevel(l)
        # reset FOM
        model.forwardModel[l] = model.generatorClass(l, kind='f')
        model.downwardModel[l] = model.generatorClass(l, kind='d')
        model.upwardModel[l] = model.generatorClass(l, kind='u')
        # reset update flags
        model.embedderTrained = False

directory = 'misure3/model_'
model_name =    ["a",   "b",    "c",    "d",    "e",    "f"]#["a",   "b",    "c",    "d",    "e",    "f",    'g',    'h', 'i',       'l']
train_model =   [2000,2000,2000,2000,2000,2000]#[0,     0,      5300,   5300,   5300,   5300,   5300,   5300, 5300,     5300]
mean_model =    [False, True, False, True, False, True]#[False, True,   False,  True,   False,  True,   False,  True, False,    True]
orth_model =    [False, False, True, True, True, True]#[False, False,  False,  False,  True,   True,   True,   True, True,     True]
lamb =          [0,0,0.1,0.1,0.01,0.01]#[0,     0,      0,      0,      0.1,    0.1,    0.01,   0.01, 0.001,    0.001]
my_seeds = [11, 33, 14556, 321321, 51243, 1231]#[5, 19, 31, 66, 75, 85, 97]

memoryLevels = 5
dist = [41000, 0.00025]
unit1 = 16
unit2 = 16
epochs = 100
batch = 10
step = 1

zeroEmbedderPretrained = True
timeStampSize = 345 # ~1/64s

model = Model(memoryLevels = memoryLevels, dist=dist, unit1=unit1, unit2=unit2, epochs=epochs, batch=batch, 
zeroEmbedderPretrained=zeroEmbedderPretrained, generatorClass=FOM3, step=step, lamb=lamb[0])

# Max elements for each song in input (only for testing)
maxItem = 0
model = Model()
model.load('model_level0_short')
model.addLevels(memoryLevels-1)
#temp = copy.deepcopy(model.embedSystem)
i = 0
tot = len(model_name)*len(my_seeds)
for my_seed in my_seeds:
    for idx in range(len(model_name)):
        #model.embedSystem = copy.deepcopy(temp)
        # random seeds must be set before importing keras & tensorflow
        i += 1
        np.random.seed(my_seed)
        random.seed(my_seed)
        tf.compat.v2.random.set_seed(my_seed)
        tf.compat.v1.set_random_seed(my_seed)
        softReset(model)
        mm = model_name[idx]
        model.updateVectorizationPass = train_model[idx]
        model.resetEmbedders(unit1, unit2, dist=dist, orthogonal=orth_model[idx], doMean=mean_model[idx], lamb=lamb[idx])
        print(" ### {}/{} ###".format(i, tot))
        print( " -- COMPUTING FOR [TRAIN {} | ORTH {} | MEAN {}] --".format(train_model[idx], orth_model[idx], mean_model[idx]))

        model.fitModelLevels()
        print("  -- END FIT --\n\n")

            
        print("Saving...")
        #model.save(directory+mm)

        print("SAVING embed data")
        #model.embedInfoToFile_small(str(dist)+" | "+str(unit1)+"-"+str(unit2)+" | ts: "+str(timeStampSize)+" | "+str(train_model[idx])+" train | "+str(mean_model[idx])+" mean vect | "+str(orth_model[idx])+" Orth | Lambda:"+str(lamb)+"", filename=directory+mm+"_info.txt")
        fl_str = "Seed: "+ str(my_seed) + " | " + str(dist)+" | "+str(unit1)+"-"+str(unit2)+" | ts: "+str(timeStampSize)+" | "+str(train_model[idx])+" train | "+str(mean_model[idx])+" mean vect | "+str(orth_model[idx])+" Orth | Lambda:"+str(lamb[idx])+""
        model.embedInfoToCSV(directory+mm+"_info_"+str(my_seed)+".csv", fl_str)
        print(model.totalChunkNumber())

print("END")


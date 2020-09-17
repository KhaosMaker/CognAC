from classes.Model import Model
from classes.MemoryLevel import MemoryLevel
from classes.Embedder import Embedder
from classes.FOM4 import FOM4
from classes.FOM3 import FOM3
from math import floor

import numpy as np
import scipy.io.wavfile as siow
import copy
import os

directory = 'esperimenti/'
model_name =    ["a",   "b",    "c",    "d",    "e",    "f",    "g",    "h",    "i"]
train_model =   [0,     0,      5000,   5000,   5000,   5000,   5000,   5000,   5000]
mean_model =    [False, True,   False,  True,   False,  True,   False,  True,   False]
orth_model =    [False, False,  False,  True,   True,   True,   True,   True,   True]
lamb =          [0,     0,      0,      0.1,    0.1,    0.01,    0.01,   0.001,  0.001]
ntry = 5

memoryLevels = 6
dist = [25000, 0.0025]
unit1 = 16
unit2 = 16
epochs = 100
batch = 10
step = 1

zeroEmbedderPretrained = True
timeStampSize = 1 # ~1/64s
cleanClasses = 10000



# Max elements for each song in input (only for testing)
i = 0
tot = len(model_name)*ntry
for t in range(ntry):
    for idx in range(len(model_name)):
        if t == 0:
            try:
                os.mkdir(directory+model_name[idx])
            except FileExistsError:
                print ("Directory {} already exist".format(directory+model_name[idx]))
            except OSError:
                print ("Creation of the directory %s failed" % directory+model_name[idx])
            else:
                print ("Successfully created the directory %s" % directory+model_name[idx])

        model = Model(memoryLevels = memoryLevels, dist=dist, unit1=unit1, unit2=unit2, epochs=epochs, batch=batch, 
            statisticalModel=FOM3, orthogonal=orth_model[idx], lamb=lamb[idx], cleanClasses=cleanClasses, 
            trainEmbedder=train_model[idx], doMean=mean_model[idx])
        model.loadFirstLayer("music_level_0")
        i += 1
        print(" ### {}/{} ###".format(i, tot))
        print( " -- COMPUTING FOR [TRAIN {} | ORTH {} | LAMB {} | MEAN {}] --".format(train_model[idx], orth_model[idx], lamb[idx], mean_model[idx]))

        model.fitModelLevels()
        print("  -- END FIT --\n\n")

        print("SAVING embed data")
        #model.embedInfoToFile_small(str(dist)+" | "+str(unit1)+"-"+str(unit2)+" | ts: "+str(timeStampSize)+" | "+str(train_model[idx])+" train | "+str(mean_model[idx])+" mean vect | "+str(orth_model[idx])+" Orth | Lambda:"+str(lamb)+"", filename=directory+mm+"_info.txt")
        fl_str = "Seed: "+ str(t) + " | " + str(dist)+" | "+str(unit1)+"-"+str(unit2)+" | ts: "+str(timeStampSize)+" | "+str(train_model[idx])+" train | "+str(mean_model[idx])+" mean vect | "+str(orth_model[idx])+" Orth | Lambda:"+str(lamb[idx])+""
        model.embedInfoToCSV(directory+model_name[idx]+"/"+model_name[idx]+"_info_"+str(t)+".csv", fl_str)

print("END")


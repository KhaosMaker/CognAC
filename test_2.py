from classes.Model import Model
from classes.MemoryLevel import MemoryLevel
from classes.Embedder import Embedder
from classes.FOM4 import FOM4
from classes.FOM3 import FOM3
from math import floor

import numpy as np
import scipy.io.wavfile as siow
import random
#import tensorflow as tf
#from tensorflow.python.compiler.tensorrt import trt_convert as trt

import glob, os

model_name = "model_PROVA"

memoryLevels = 5
dist = [28000, 0.0025]#
unit1 = 16
unit2 = 8
epochs = 200
batch = 2
step = 1
doMean = True
orthogonal = True
lamb = 0.001
zeroEmbedderPretrained = True
trainEmbedder = 50
cleanClasses= 10000
timeStampSize = 300 # ~1/64s

model = Model(memoryLevels = memoryLevels, dist=dist, unit1=unit1, unit2=unit2, epochs=epochs, batch=batch, 
statisticalModel=FOM3, orthogonal=orthogonal, lamb=lamb, cleanClasses=cleanClasses, trainEmbedder=trainEmbedder)
model.loadFirstLayer("firstLayer_backup")
model.fitModelLevels()
print("  -- END FIT --\n\n")
model.printModel(max=40)

print("____________________________")
#s = "ML: {} | dist: {} | unit1: {} | unit2: {} | TS: {}".format(memoryLevels, dist, unit1, unit2, timeStampSize)
#model.ebedInfoToFile(s, "model_1")

os.chdir(direct)
print("Saving...")
model.save(model_name)

print("Generating!")	
model.generateSong(model_name+"_out_normal.wav", 300, 0, samplerate)
model.generateSong_firstOrder(model_name+"_out_firstOrder.wav", start=0, n=800, samplerate=samplerate)
model.generate_freewheel(model_name+"_freewheel.wav", start=1, n=1900, samplerate=samplerate)

print("SAVING embed data")
#model.embedInfoToFile("[40000, 0.00035] | 16-16 | ts: 345 | 4.8 train | no mean vect | no Orth", filename=model_name+"_info.txt")

print(model.totalChunkNumber())

print("END")
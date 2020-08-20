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

#songname = ["song_pkmn_1.wav", "song_pkmn_2.wav", "song_pkmn_3.wav", "song_pkmn_4.wav"]#["song1.wav"]#,"song2.wav","song3.wav","song4.wav"]#, "song_porta.wav"]
songname = []
songFolder = "/WAVS"
direct = os.getcwd()
os.chdir(os.getcwd()+songFolder)
for file in glob.glob("*.wav"):
    songname.append(songFolder+file)

random.shuffle(songname)
os.chdir(direct)

model_name = "model_PROVA"

memoryLevels = 9
dist = [5500, 0.035]#
unit1 = 16 
unit2 = 8
epochs = 200
batch = 10
step = 1
doMean = True
orthogonal = True
lamb = 0.001
zeroEmbedderPretrained = True
_updateVectorizationPass = 9800
cleanClasses=10000
timeStampSize = 250 # ~1/64s

model = Model(memoryLevels = memoryLevels, dist=dist, unit1=unit1, unit2=unit2, epochs=epochs, batch=batch, 
zeroEmbedderPretrained=zeroEmbedderPretrained, generatorClass=FOM3, step=step, orthogonal=orthogonal, lamb=lamb, cleanClasses=cleanClasses)


# Max elements for each song in input (only for testing)
maxItem = 0
model = Model()
model.load('model_PROVA')
model.addLevels(memoryLevels-1)
model.fitModelLevels()
print("  -- END FIT --\n\n")
model.printModel(max=40)

print("____________________________")
#s = "ML: {} | dist: {} | unit1: {} | unit2: {} | TS: {}".format(memoryLevels, dist, unit1, unit2, timeStampSize)
#model.ebedInfoToFile(s, "model_1")

print("Saving...")
model.save(model_name)

print("Generating!")	
model.generateSong(model_name+"_out.wav", 200, 0, samplerate)

print("SAVING embed data")
#model.embedInfoToFile("[40000, 0.00035] | 16-16 | ts: 345 | 4.8 train | no mean vect | no Orth", filename=model_name+"_info.txt")

print(model.totalChunkNumber())

print("END")
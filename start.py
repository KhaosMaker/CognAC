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

songname = []
direct = os.getcwd()
os.chdir(os.getcwd()+"/WAVS")
for file in glob.glob("*.wav"):
    songname.append(file)
#songname = ["song_pkmn_1.wav"]
random.shuffle(songname)
model_name = "model_PROVA"

memoryLevels = 9
dist = [22000, 0.0002]#
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
cleanClasses= 10000
timeStampSize = 250 # ~1/64s

model = Model(memoryLevels = memoryLevels, dist=dist, unit1=unit1, unit2=unit2, epochs=epochs, batch=batch, 
statisticalModel=FOM3, orthogonal=orthogonal, lamb=lamb, cleanClasses=cleanClasses)


# Max elements for each song in input (only for testing)
maxItem = 0
for i in range(3):
	random.shuffle(songname)
	for idx in range(len(songname)):
		song = songname[idx]
		# Read the file
		print("\t|| {} / {} ||".format(idx+1, len(songname)))
		print("  -- COMPUTING: ", song," --")
		samplerate, data = siow.read(song)
		data = data[:,0]

		t = np.zeros((data.shape[0]))
		v = 100
		t = data/v

		t = t.astype(np.int32)
		t = t*v
		ts = timeStampSize
		n = int(t.shape[0]/ts)
		t = t[:n*ts]
		t = t.reshape(n, ts)
		if maxItem > 0:
			t = t[random.randint(0,5):maxItem]
		model.updateVectorizationPass = _updateVectorizationPass

		print("  --  STARTING MODEL  --")
		#model.resetMemory()
		model.data = t
		
		#model.getBatchEmbedding()
		
		model.fitDataLevel()
		model.clean()


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
model.generateSong(model_name+"_out.wav", 300, 0, samplerate)

print("SAVING embed data")
#model.embedInfoToFile("[40000, 0.00035] | 16-16 | ts: 345 | 4.8 train | no mean vect | no Orth", filename=model_name+"_info.txt")

print(model.totalChunkNumber())

print("END")
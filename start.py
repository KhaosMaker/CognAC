from classes.Model import Model
from classes.MemoryLevel import MemoryLevel
from classes.Embedder import Embedder
from classes.FOM4 import FOM4
from classes.FOM3 import FOM3
from math import floor

import numpy as np
import scipy.io.wavfile as siow
#import tensorflow as tf
#from tensorflow.python.compiler.tensorrt import trt_convert as trt

songname = ["song1.wav","song2.wav","song3.wav","song4.wav"]#, "song_porta.wav"]
model_name = "model_PROVA"

memoryLevels = 5
dist = [41000, 0.00025]#
unit1 = 16
unit2 = 16
epochs = 100
batch = 10
step = 1
doMean = True
orthogonal = True
lamb = 0.01
zeroEmbedderPretrained = True
_updateVectorizationPass = 4800
timeStampSize = 550 # ~1/64s

model = Model(memoryLevels = memoryLevels, dist=dist, unit1=unit1, unit2=unit2, epochs=epochs, batch=batch, 
zeroEmbedderPretrained=zeroEmbedderPretrained, generatorClass=FOM3, step=step, orthogonal=orthogonal, lamb=lamb)

# Max elements for each song in input (only for testing)
maxItem = 4000

for song in songname:
	# Read the file
	samplerate, data = siow.read(song)
	data = data[:,0]

	t = np.zeros((data.shape[0]))
	v = 1
	t = data/v

	t = t.astype(np.int32)
	t = t*v
	ts = timeStampSize
	n = int(t.shape[0]/ts)
	t = t[:n*ts]
	t = t.reshape(n, ts)
	if maxItem > 0:
		t = t[:maxItem]
	model.updateVectorizationPass = _updateVectorizationPass

	print("  -- COMPUTING: ", song," --")
	print("  --  STARTING MODEL  --")
	#model.resetMemory()
	model.data = t
	
	#model.getBatchEmbedding()
	
	model.fitDataLevel()
	model.clean()

model.fitModelLevels()
print("  -- END FIT --\n\n")
#model.printModel(max=40)

print("____________________________")
#s = "ML: {} | dist: {} | unit1: {} | unit2: {} | TS: {}".format(memoryLevels, dist, unit1, unit2, timeStampSize)
#model.ebedInfoToFile(s, "model_1")

	
print("Saving...")
model.save(model_name)

print("Generating!")	
model.generateSong(model_name+"_out.wav", 140250, 0, samplerate)

print("SAVING embed data")
#model.embedInfoToFile("[40000, 0.00035] | 16-16 | ts: 345 | 4.8 train | no mean vect | no Orth", filename=model_name+"_info.txt")

print(model.totalChunkNumber())

print("END")
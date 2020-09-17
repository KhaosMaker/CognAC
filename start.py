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

memoryLevels = 6
dist = [0, 0.0015]#
unit1 = 16
unit2 = 16
epochs = 200
batch = 10
step = 1
doMean = True
orthogonal = True
lamb = 0.001
zeroEmbedderPretrained = True
trainEmbedder = 10000
cleanClasses= 10000
timeStampSize = 1 # ~1/64s

model = Model(memoryLevels = memoryLevels, dist=dist, unit1=unit1, unit2=unit2, epochs=epochs, batch=batch, 
statisticalModel=FOM3, orthogonal=orthogonal, lamb=lamb, cleanClasses=cleanClasses, trainEmbedder=trainEmbedder)


# Max elements for each song in input (only for testing)
maxItem = 4000
for i in range(1):
	random.shuffle(songname)
	for idx in range(len(songname)):
		song = songname[idx]
		# Read the file
		print("\t|| {} / {} ||".format(idx+1, len(songname)))
		print("  -- COMPUTING: ", song," --")
		samplerate, data = siow.read(song)
		data = data[:,0]
		data = data[::2] #half the number of samples
		samplerate = floor(samplerate/2) # half the samplerate

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
			t = t[:maxItem]

		print("  --  STARTING MODEL  --")
		
		model.fitDataLevel(t)#[random.randint(0,40):])
		#model.clean()

os.chdir(direct)
model.saveFirstLayer("music_level_0")
exit()
#input()


model.fitModelLevels()
print("  -- END FIT --\n\n")
model.printModel(max=40)

print("____________________________")
#s = "ML: {} | dist: {} | unit1: {} | unit2: {} | TS: {}".format(memoryLevels, dist, unit1, unit2, timeStampSize)
#model.ebedInfoToFile(s, "model_1")


print("Saving...")
model.save(model_name)

print("Generating!")	
#model.generateSong(model_name+"_out_normal.wav", 300, 0, samplerate)
#model.generateSong_firstOrder(model_name+"_out_firstOrder.wav", start=0, n=800, samplerate=samplerate)
model.generateFreewheel(model_name+"_freewheel.wav", start=1, n=1900, samplerate=samplerate)

print("SAVING embed data")
#model.embedInfoToFile("[40000, 0.00035] | 16-16 | ts: 345 | 4.8 train | no mean vect | no Orth", filename=model_name+"_info.txt")

print(model.totalChunkNumber())

print("END")
from classes.Model import Model
from classes.MemoryLevel import MemoryLevel
from classes.Embedder import Embedder
from classes.FOM4 import FOM4
from classes.FOM3 import FOM3
from math import floor

import numpy as np
import scipy.io.wavfile as siow
import random

import glob, os
model_name = "model_char_short_5"

memoryLevels = 6
dist = [0, 0.015]#
unit1 = 16
unit2 = 16
epochs = 200
batch = 10
step = 1
doMean = True
orthogonal = True
lamb = 0.001
zeroEmbedderPretrained = True
trainEmbedder = 3000
cleanClasses= 10000

model = Model(memoryLevels = memoryLevels, dist=dist, unit1=unit1, unit2=unit2, epochs=epochs, batch=batch, 
statisticalModel=FOM3, orthogonal=orthogonal, lamb=lamb, cleanClasses=cleanClasses, trainEmbedder=trainEmbedder)

data = []
with open("emma.txt", "r") as f:
    for line in f:
        for c in line:
            if c in [" ", "-", "\n", "\t", "*", "_"]:
                continue
            else:
                data.append([ord(c.lower())])

print("  --  STARTING MODEL  --")
model.fitDataLevel(data[1:50000])



model.fitModelLevels()
print("  -- END FIT --\n\n")
model.printModel(max=40)
model.save(model_name)

print("END")

#1: 0.00025, train 10k, 6  layer, window 5, d 1.75 (all)
"""
EmbedSystem 1: [1064] [10.25852168824519 +/- 4.61960109517453]
EmbedSystem 2: [986] [5.066345823306978 +/- 3.412947829235754]
EmbedSystem 3: [927] [4.304318212777551 +/- 3.554445217687118]
EmbedSystem 4: [647] [1.8509680324178297 +/- 1.8967577331065932]
"""
#2: 0.0025, train5k, 6 layer, window 7, d 2
"""
EmbedSystem 1: [1102] [13.099217785096748 +/- 7.106009425954828]
EmbedSystem 2: [877] [10.779342723004694 +/- 8.918133473866797]
EmbedSystem 3: [140] [10.004854368932039 +/- 7.78397200808113]
EmbedSystem 4: [14] [8.117647058823529 +/- 7.80930817151219]
"""
# 3: 0.00015, train 5k, 6layer, w10, d 1 (100k el)
"""
EmbedSystem 1: [1128] [10.410096098182239 +/- 4.624524306373649]
EmbedSystem 2: [1161] [7.271211670139922 +/- 4.845004616562852]
EmbedSystem 3: [674] [5.269187986651835 +/- 4.515853848041488]
EmbedSystem 4: [291] [2.6720867208672088 +/- 3.5314281864335446]
"""
# 4: 0.0015, train 5k, 6layer, w10, d 1 (100k el)
"""
EmbedSystem 1: [1128] [10.409348605808168 +/- 4.626325121284009]
EmbedSystem 2: [1161] [7.195581737849779 +/- 4.846133807824901]
EmbedSystem 3: [674] [5.331838565022421 +/- 4.600758218967577]
EmbedSystem 4: [291] [2.7577464788732393 +/- 3.603657196485643]
"""
# 5: 0.015,  train 5k, 6layer, w10, d 1 (30k el)
"""
EmbedSystem 1: [17] [8.138610182088444 +/- 4.381046607204245]
EmbedSystem 2: [56] [7.147229114971051 +/- 4.841681712235463]
EmbedSystem 3: [25] [6.003215434083601 +/- 4.429008767194986]
EmbedSystem 4: [9] [4.52054794520548 +/- 4.340089569739751]
"""
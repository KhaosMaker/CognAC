from classes.MemoryLevel import MemoryLevel
from classes.FOM3 import FOM3
from classes.EmbedSystem import EmbedSystem
from classes.Memory import Memory
from tqdm import tqdm 
import numpy as np
import scipy.io.wavfile as siow
import json
import random
import operator
import os
import pickle

from time import time
from math import floor
import random
import csv
from scipy.sparse import coo_matrix
import copy


class Model:

    def __init__(self, data=[], memoryLevels=1, statisticalModel=FOM3, trainEmbedder=0,
                    dist=[1,1], unit1=1, unit2=1, special_c= -45000, epochs=10, batch=1, maxlen=0,
                    orthogonal=False, doMean=False, lamb=0.01, cleanClasses=5000):
        
        self.levels = memoryLevels
        self.memory = Memory(self.levels)
        self.forwardModel = []
        self.upwardModel = []
        self.downwardModel = []
        self.embedSystem = []
        self.statisticalModel = statisticalModel
        self.trainEmbedder = trainEmbedder
        self.epochs = epochs
        self.batch = batch
        self.embedderTrained = False
        # First order models + EmbedSystems       
        for ml in range(memoryLevels):
            self.forwardModel.append(self.statisticalModel(ml, kind='f'))
            self.upwardModel.append(self.statisticalModel(ml,  kind='u'))
            self.downwardModel.append(self.statisticalModel(ml, kind='d'))

        self.embedSystem.append(EmbedSystem(0, dist[0], unit1, unit2, special_c, doMean=doMean, orthogonal=orthogonal, lamb=lamb))
        self.embedSystem.append(EmbedSystem(1, dist[1], unit1, unit2, special_c, doMean=doMean, orthogonal=orthogonal, lamb=lamb))
        for ml in range(2, memoryLevels):
            self.embedSystem.append(EmbedSystem(ml, dist[1], embedder=self.embedSystem[1].emb, doMean=doMean, orthogonal=orthogonal))

        self.data = data
        self.__genC = statisticalModel
        self.count = 0
        self.statisticalModel = statisticalModel

        self.unit1 = unit1
        self.unit2 = unit2
        self.special_c = special_c
        self.dist = dist
        self.maxlen = maxlen
        self.doMean = doMean
        self.orthogonal = orthogonal
        self.lamb = lamb

        self.cleanClasses = cleanClasses

   
    def fit(self, reset=0, saveSteps=0, savename='model.json'):
        """
        Get the input
        pass to the embedSystem
        Get the class
        push the class in memory
        update FOM
        """
        # If we do not need to update the RNN we can free the memory
        if self.totalChunkNumber() > self.trainEmbedder:
            self.memory.resetMemory()
        self.fitDataLevel()
        self.fitModelLevels()      


    def fitDataLevel(self):
        """
        fit the level 0
        """
        print("DATA LEVEL: ")
        # For each ground elemente
        #for d in self.data:
        for idx in tqdm(range(len(self.data))):
            #if self.memory.memory[0].memoryLength % self.cleanClasses*10 == 0:
            #    self.clean()
            # START getting input
            d = np.array(self.data[idx])
            c = self.embedSystem[0].computeClass(d, dataLevel=True, inlist=True)
            
            self.memory.addToMemory(0, c)

            #update the forward model
            self.forwardModel[0].update(
                self.memory.beforeItem(0), 
                self.memory.actualItem(0)
                )

    
    def fitModelLevels(self):
        """
        fit levels from 1 to <levels>-1
        """
        print("UPPER LEVELS: ")
        # For each level (the zero-th too) prepare to chunk
        for idx in tqdm(range(1, self.memory.getLevel(0).memoryLength)):
            actualIndex = idx
            if idx%self.cleanClasses == 0:
                self.clean()
            if self.updateEmbedder():
                self.embedderTrained = True
                for l in range(1, self.levels):
                    print("\nFITTING LSTM LEVEL {}: ".format(l))  
                    self.embedSystem[l].autofit(self.epochs, self.batch)
                    #print("\n______________________________\n")
                    #print("|| MEMORY CONSOLIDATION ({}) ||".format(l))
                    #self.consolidateMemoryLevel(l)
                    self.embedSystem[l].vd.reset()
                    self.memory.resetMemoryLevel(l)
                    self.forwardModel[l] = self.statisticalModel(l, kind='f')
                    self.downwardModel[l] = self.statisticalModel(l, kind='d')
                    self.upwardModel[l] = self.statisticalModel(l, kind='u')
                self.fitModelLevels()
                return
                print("Memory consolidation end")
                print()

            for i in range(self.levels-1):
                # IF the memory has to be chunked after the new insterion
                condition = self.memory.hasToChunk(i, self.forwardModel, self.embedSystem, 
                actualIndex-1, self.memory.getLevel(i).getItem(actualIndex))
                if condition:
                    # Reset state of the embedder
                    self.embedSystem[i].resetPartialState()
                    # create the new chunk and return it as <element>, not ID
                    newChunk = self.memory.chunk(i+1, actualIndex-1)  
                    chunkClass = self.embedSystem[i+1].computeClass(newChunk)   
                    self.memory.addToMemory(i+1, chunkClass)

                    # update the forward model at level i+1                        
                    self.forwardModel[i+1].update(self.memory.beforeItem(i+1), self.memory.actualItem(i+1))                    

                    # Update the upward model with the starting symbol of the chunk and the new chunk
                    #chunkString = self.memory.chunkToString(newChunk, i+1)
                    self.upwardModel[i].update(newChunk[0], chunkClass)

                    # Update the downward model with the transition between the chunk and its last symbol
                    # NB the downward model is the one of the level ABOVE.
                    self.downwardModel[i+1].update(chunkClass, newChunk[len(newChunk)-1])
                    
                    actualIndex = self.memory.getLevel(i+1).memoryLength-1

                else:
                    break

    def printModel(self, max=10, start=0):
        for i in range(self.levels):
            print()
            print("# ================================= #")
            self.memory.getLevel(i).printLevel(max, start)
        print()

    def printEmbeddingInfo(self):
        res = ""
        for i in range(self.levels):
            res += self.embedSystem[i].strClassInfo()
        with open("embedInfo.txt", "w") as f:
            f.write(res)
    
    def embedInfoToFile(self, s="", filename="info.txt"):
        res = s+"\n"
        for i in range(1, self.levels):
            res += self.embedSystem[i].strClassInfo()
        with open(filename, "w") as f:
            f.write(res)
    
    def embedInfoToFile_small(self, s="", filename="info.txt"):
        res = s+"\n"
        for i in range(1, self.levels):
            res += self.embedSystem[i].strClassInfo_small()
        with open(filename, "w") as f:
            f.write(res)
    
    def embedInfoToCSV(self, filename, flStr=None):
        res = []
        for i in range(1, self.levels):
            res.append(self.embedSystem[i].csvClassInfo())
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            if flStr is not None:
                writer.writerow([flStr])
            writer.writerow(['level', 'Class dim', 'Var','Mean Dist', 'Var'])
            for element in res:
                writer.writerow(element)


    def resetMemory(self):
        self.memory.resetMemory()

    def changeMean(self, value):
        for idx in range(self.levels):
            self.embedSystem[idx].vd.doMean = value
    
    def resetEmbedders(self, unit1, unit2, dist, special_c= -45000, orthogonal=False, doMean=False, lamb=0.1):
        self.embedSystem = [self.embedSystem[0]]
        self.embedSystem.append(EmbedSystem(1, dist[1], unit1, unit2, special_c, doMean=doMean, orthogonal=orthogonal, lamb=lamb))
        for ml in range(2, self.levels):
            self.embedSystem.append(EmbedSystem(ml, dist[1], embedder=self.embedSystem[1].emb, doMean=doMean, orthogonal=orthogonal))
        
                
    def totalChunkNumber(self):
        return sum([self.embedSystem[i].uniqueChunkNumber() for i in range(1, self.levels)])
    
    def updateEmbedder(self):
        #print("---> {} >= {}".format(self.totalChunkNumber(), self.trainEmbedder))
        return not self.embedderTrained and self.totalChunkNumber() >= self.trainEmbedder


    def getBatchEmbedding(self):
        self.embedSystem[0].fit(self.data, epochs=self.epochs, batch=self.batch)
        embeddingList = self.embedSystem[0].getBatchEmbedding(self.data)
        l1 = []
        for i in tqdm(range(len(self.data))):
            cl = self.embedSystem[0].computeClass(self.data[i], dataLevel=True)#embeddingList[i], self.data[i])
            l1.append(cl)
        print(len(l1), sum(l1))
        self.data = l1

        print("DATA LEVEL: ")
        # For each ground elemente
        #for d in self.data:
        for d in tqdm(self.data):
            # START getting input
            #d = np.array(d)
            #c = self.embedSystem[0].computeClass(d, dataLevel=True, inlist=True)
            
            self.memory.addToMemory(0, d)

            #update the forward model
            self.forwardModel[0].update(
                self.memory.beforeItem(0), 
                self.memory.actualItem(0)
                )
        #self.fitModelLevels()
    
    def addLevel(self):
        self.levels += 1
        self.forwardModel.append(self.statisticalModel(self.levels-1, kind='f'))
        self.upwardModel.append(self.statisticalModel(self.levels-1,  kind='u'))
        self.downwardModel.append(self.statisticalModel(self.levels-1, kind='d'))
        if self.levels <= 2:
            self.embedSystem.append(EmbedSystem(self.levels-1, self.dist[min(1, max(self.levels-1, 0))], self.unit1, self.unit2, self.special_c, self.maxlen, doMean=self.doMean, orthogonal=self.orthogonal, lamb=self.lamb))
        else:
            self.embedSystem.append(EmbedSystem(self.levels-1, self.dist[1], maxlen=self.maxlen, embedder=self.embedSystem[0].emb, doMean=self.doMean, orthogonal=self.orthogonal))
        self.memory.addMemoryLevel()
    
    def addLevels(self, n):
        for i in range(n):
            self.addLevel()


     
    # ============ #
    #  GENERATION  #
    # ============ #

    def generateSong(self, filename, n, sl, samplerate):
        generation = self.generate(n, startingLevel=sl)
        generation = list(map(int, generation))
        generation = np.array(generation)
        generation = generation.astype(np.int16)
        print("GENERATION: ")
        siow.write(filename, samplerate, generation)  
    
    def generateSong_firstOrder(self, filename, start, n, samplerate):
        generation = self.generate_firstOrder(start=start, n=n)
        generation = list(map(int, generation))
        generation = np.array(generation)
        generation = generation.astype(np.int16)
        print("GENERATION: ")
        siow.write(filename, samplerate, generation)  

    def generate_freewheel(self, filename, start, n, samplerate):
        generation = self.freeWheel(start=start, n=n)
        generation = list(map(int, generation))
        generation = np.array(generation)
        generation = generation.astype(np.int16)
        print("GENERATION: ")
        siow.write(filename, samplerate, generation)

    def startGenerate(self):
        """
        Start of the generation one elements.
        """
        givenLevel = self.getStartingLevel()
        givenToken = self.memory.getFringeToken(givenLevel)
        gen = self.generateToken(givenToken, givenLevel)
        print("From Level ", givenLevel, " take ", givenToken, " result in -> ", gen)
        return gen

    def generate(self, n, start=None, startingLevel=None):
        if startingLevel is None:
            givenLevel = self.getStartingLevel()
        else:
            givenLevel = startingLevel
        #TODO -> change first token!
        if start is None:
            givenToken = self.memory.getLevel(givenLevel).memory[0]
        else:
            givenToken = start
        
        gen, givenLevel = self.generateToken(givenToken, givenLevel)
        actualLevel = givenLevel
        result = []
    
        totalLen = []
        for _ in tqdm(range(n)):
            #print(actualLevel, end="->")
            #print("From Level ", givenLevel, " take ", givenToken, " result in -> ", gen) 
            # IF there is no change in the level of the prediction, add the prediction 
            if (actualLevel - givenLevel) == 0:
                addingRes = self.translateClass(gen, givenLevel)
                #print("MODEL GEN: {} -> {} (LV.{})".format(gen,addingRes, givenLevel))
                result = result + addingRes
                totalLen.append(len(addingRes))
            elif (actualLevel - givenLevel) < 0:
                # IF the prediction is in the layer above, overwrite the last prediction
                # (that happen in the layer below, it was the start of the chunk)
                addingRes = self.translateClass(gen, givenLevel)
                result = result[:-1]
                result += addingRes
                totalLen[-1] = len(addingRes)
            # if there is a downward, do nothing.
            actualLevel = givenLevel     
            givenToken = gen
            
            gen, givenLevel = self.generateToken(givenToken, givenLevel)
           
        print("TOTAL LEN: {}".format(sum(totalLen)))
        print("MEAN LEN: {}".format(sum(totalLen)/n))
        print("Max: {} [{}%]".format(max(totalLen), max(totalLen)/sum(totalLen)*100))

            
        return result

            
    def translateClass(self, c, level):
        """
        transform a class in a sequence of single item
        """
        if level < 0:
            return [c]
        poss, counts = self.embedSystem[level].getValueList(c)
        # Choose one random element from the possibilities
        choice = random.choices(poss, weights=counts)[0]
        res = []
        for el in choice:
            res += self.translateClass(el, level-1)
        return res


    def getStartingLevel(self):
        """
        Return the level where to start the generation, using a
        aheuristic. (highest h)
        """
        # TODO -> Find a good way
        return 0
        res = []
        for l in range(self.levels):
            t = self.memory.getFringeh(l)
            if t is not None:
                res.append(t)
        return res.index(max(res))
    
    
    def generateToken(self, token, level):
        """
        Generate the next token given the distribution probability
        """
        # Get a normalized transition distribution
        # T -> Possible transitions
        # D -> Distribution over T
        # L -> Levels
        T, D, L = self.getDistribution(token, level)

        # If the last item in the mem is a new token, the Transition Matrix row is empty
        if len(T) == 0:
            res = random.choices([x for x in self.embedSystem[level].vd.classList.keys()])[0]
            return res, level
        
        # Get the random element
        c = random.choices(T, weights=D)[0]
        return c, L[T.index(c)]


    
    
    def getDistribution(self, token, level):
        """
        Return:
            - first the element array
            - second the normalized distribution of the transition from <token> in <level>
        """
        # Forward
        fc, fd = self.forwardModel[level].getDistribution(token)
        # Downward
        dc, dd = self.downwardModel[level].getDistribution(token)
        # Upward        
        uc, ud = self.upwardModel[level].getDistribution(token)

        #fd = [x**2 for x in fd]
        fl = [level]*len(fc)

        #dd = [x**2 for x in dd]
        dl = [level-1]*len(dc)
     
        #ud = [x**2 for x in ud]
        ul = [level+1]*len(uc)

        # total distribution
        tc = fc + dc + uc
        td = self.normalize(fd + dd + ud)
        tl = fl + dl + ul
        
        # DEBUG vv
        #print(token, " | TC: ", tc)        
        # DEBUG ^^
        return tc, td, tl
    

    def normalize(self, distr):
        """
        Given an array of float (probabilities distribution)
        return the array normalized (sum up to 1)
        """
        tot = sum(distr)
        return [e/tot for e in distr]


    def computeMean_h(self):
        total = 0
        for _ in range(self.levels):
            for memory in self.memory.memory:
                if len(memory.h) > 0:
                    total += sum(memory.h) / len(memory.h)
        return total / self.levels



    def clean(self):
        print("CLEANING...")
        for i in range(len(self.embedSystem)):
            if len(self.embedSystem[i].vd.classEmb) > 1000:
                print("Cleaning ES ", i)
                v = self.embedSystem[i].clean()
                print("From {} to {} (-{})".format(len(self.embedSystem[i].vd.classEmb)+len(v), len(self.embedSystem[i].vd.classEmb), len(v)))
                for c in v:
                    #print("delete ", c, "from FM (",i,")")
                    #self.forwardModel[i].cleanFOM(c)
                    continue
            else:
                print("EmbedSystem ", i," of size [", len(self.embedSystem[i].vd.classEmb), "] is too small.")
    # SAVING MODULE
    #____________________________________________________

    def save(self, directory="model"):
        try:
            os.mkdir(directory)
        except FileExistsError:
            print ("Directory {} already exist".format(directory))
        except OSError:
            print ("Creation of the directory %s failed" % directory)
        else:
            print ("Successfully created the directory %s" % directory)
        
        self.saveModel(directory)
        self.saveEmbedSystem(directory)
    

    def saveModel(self, directory):
        modelPath = directory+"/model.pk"
        modelDict = {}
        modelDict["levels"] = self.levels
        modelDict["memory"] = self.memory
        modelDict["forwardModel"] = self.forwardModel
        modelDict["upwardModel"] = self.upwardModel
        modelDict["downwardModel"] = self.downwardModel
        modelDict["epochs"] = self.epochs
        modelDict["batch"] = self.batch
        modelDict["embedderTrained"] = self.embedderTrained
        with open(modelPath, "wb") as f:
            pickle.dump(modelDict, f)
    
    def saveEmbedSystem(self, directory):
        vdDict = {}
        for i in range(self.levels):
            self.embedSystem[i].emb.save(directory)
            vdDict[i] = self.embedSystem[i].vd.getSave()
        
        with open(directory+"/vector_dictionary.pk", "wb") as f:
            pickle.dump(vdDict, f)

    def saveFirstLayer(self, directory):
        try:
            os.mkdir(directory)
        except FileExistsError:
            print ("Directory {} already exist".format(directory))
        except OSError:
            print ("Creation of the directory %s failed" % directory)
        else:
            print ("Successfully created the directory %s" % directory)
        modelPath = directory+"/layer0.pk"
        modelDict = {}
        modelDict["layer0"] = self.memory.memory[0]
        modelDict["forwardModel"] = self.forwardModel[0]
        modelDict["VectorDict"] = self.embedSystem[0].vd.getSave()
        with open(modelPath, "wb") as f:
            pickle.dump(modelDict, f)

    def loadFirstLayer(self, directory):
        with open(directory+"/layer0.pk", "rb") as f: 
            modelDict = pickle.load(f)
        self.memory.memory[0] = modelDict["layer0"]
        self.forwardModel[0] = modelDict["forwardModel"]
        self.embedSystem[0].vd.load(modelDict["VectorDict"])


    def load(self, directory):
        self.loadModel(directory)
        self.loadEmbedSystem(directory)

    
    def loadModel(self, directory):
        with open(directory+"/model.pk", "rb") as f: 
            modelDict = pickle.load(f)
        
        self.levels = modelDict["levels"]
        self.memory = modelDict["memory"]
        self.forwardModel = modelDict["forwardModel"]
        self.downwardModel = modelDict["downwardModel"]
        self.upwardModel = modelDict["upwardModel"]
        self.epochs = modelDict["epochs"] 
        self.batch = modelDict["batch"]
        self.embedderTrained = modelDict["embedderTrained"]

    
    def loadEmbedSystem(self, directory):
        with open(directory+"/vector_dictionary.pk", 'rb') as f:
            vdDict = pickle.load(f)
        
        self.embedSystem = []
        print("  --  Loading Embed System LV 0  --")
        # LV 0
        es = EmbedSystem(0)
        es.emb.load(directory, 0)
        es.vd.load(vdDict[0])
        self.embedSystem.append(es)

        if self.levels > 1:
            # LV 1
            print("  --  Loading Embed System LV 1  -- ")
            es = EmbedSystem(1)
            es.emb.load(directory, 1)
            es.vd.load(vdDict[1])
            self.embedSystem.append(es)
            # OTHERS
            for l in range(2, self.levels):
                print("  --  Loading Embed System LV", l,"  --")
                es = EmbedSystem(l)
                es.emb = self.embedSystem[1].emb
                es.vd.load(vdDict[l])
                self.embedSystem.append(es)

    #____________________________________________________

    def generate_firstOrder(self, start = 0, n = 50):
        result = []
        result = result + self.translateClass(start, 0)
        token = start
        for i in tqdm(range(n)):
            possibleTokens = self.forwardModel[0].getNext(token)
            prob = []
            for p in possibleTokens:
                prob.append(self.forwardModel[0].getProbability(token, p))
            if len(prob) == 0:
                choice = start
            else:
                choice = random.choices(possibleTokens, weights=prob)[0]
            element = self.translateClass(choice, 0)
            token = choice
            result = result + element
        return result

    def freeWheel(self, start=0, n=100):
        memory = Memory(self.levels)
        memory.addToMemory(0, start)
        token = start        
        for idx in tqdm(range(n)):
            possibleTokens = self.forwardModel[0].getNext(token)
            print("[{}] -> [{}]".format(token, possibleTokens))
            if len(possibleTokens) == 0:
                possibleTokens = self.forwardModel[0].getRandomToken()
                input()
            prob = []
            for data in possibleTokens:
                #print("> {}".format(data))
                memory.addToMemory(0, data)
                cleanlevels = 1
                actualIndex = idx
                for i in range(self.levels-1):
                    # IF the memory has to be chunked after the new insterion
                    condition = memory.hasToChunk(i, self.forwardModel, self.embedSystem, 
                    actualIndex-1, data, influence=True)
                    if condition:

                        # Reset state of the embedder
                        self.embedSystem[i].resetPartialState()
                        # create the new chunk and return it as <element>, not ID
                        newChunk = memory.chunk(i+1, actualIndex-1)  
                        #print("{} |{} | {}".format(i, actualIndex, newChunk))
                        #print("{}".format(memory.getLevel(i).memoryLength))
                        #print("{}".format(memory.getLevel(i).memory))
                        #input()
                        chunkClass = self.embedSystem[i+1].computeClass(newChunk, push=False)   
                        memory.addToMemory(i+1, chunkClass)
                        
                        actualIndex = memory.getLevel(i+1).memoryLength-1
                        cleanlevels = i+1

                    else:
                        break
                prob.append(memory.computeProbability(0, self.forwardModel, idx-1, data, self.embedSystem))
                for l in range(cleanlevels):
                    memory.getLevel(l).deleteLast()
            #print("FROM: ", token, ":")
            #ads = [(possibleTokens[i], prob[i]) for i in range(len(possibleTokens))]
            #for e in ads:
            #    print(e)
            token = random.choices(possibleTokens, weights=prob)[0]
            #print(token, end=" -> ")
            memory.addToMemory(0, token)
            #print("TOKEN -> ", token)
        result = []
        for token in memory.getLevel(0).memory:
            result = result + self.translateClass(token, 0)
        return result





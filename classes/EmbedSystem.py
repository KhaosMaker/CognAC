from classes.Embedder import Embedder
from classes.VectorDict import VectorDict

import numpy as np
from numpy import mean, std

from time import time

class EmbedSystem():
    def __init__(self, level, dist=10, unit1=1, unit2=1, special_c= -45000, embedder=None, doMean=False, orthogonal=False, lamb=0.01):
        self.vd = VectorDict(dist, level=level, doMean=doMean)
        if embedder is not None:
            self.emb = embedder
        else:
            self.emb = Embedder(level, unit1, unit2, special_c, orthogonal=orthogonal, lamb=lamb)
        self.level = level

    
    def getNearClass(self, tokenSeq):
        embseq = self.emb.get_embedding(tokenSeq)
        res = self.vd.getNearClass(embseq)
        return res
    
    def computeClass(self, tokenSeq, precomputation=True, push=True, inlist=False, dataLevel=False):
        """
        tokenSeq :- input sequence
        precoputation :- if allow the precomputation of the token
        push :- if push the tokenSeq in chunkList
        inlist :- append precomputation to to chunkList
        return :- class

        NB: if class do not exist, it will be crated
        """
        # ? vvvvvvvvvvvvv
        if dataLevel:
            if precomputation:
                pre = self.vd.preComputeVector(tokenSeq)
                if pre is not None:
                    #print("YES!")
                    if inlist:
                        self.vd.insertSeq(tokenSeq, tokenSeq, pre, push)
                    return pre
            
            #embseq = tokenSeq#self.emb.get_embedding(tokenSeq)
            res = self.vd.computeClass(tokenSeq, tokenSeq, push, dataLevel=dataLevel) #self.vd.addNewClass(tokenSeq, tokenSeq, push)
            return res
        # ? ^^^^^^^^^^^^^^^

        if precomputation:
            pre = self.vd.preComputeVector(tokenSeq)
            if pre is not None:
                if inlist:
                    self.vd.insertSeq(tokenSeq, tokenSeq, pre, push)
                return pre

        embseq = self.emb.get_embedding(tokenSeq)
        return self.vd.computeClass(embseq, tokenSeq, push)

    def getClass(self, seq):
        pre = self.vd.preComputeVector(seq)
        if pre is not None:
            return pre
        emb = self.emb.get_embedding(seq)
        return self.vd.getOnlyClass(emb, seq)
    
    def getNewClassId(self):
        return self.vd.actualId



    
    def getValueList(self, c):
        """
        c :- class of elements
        return :- list of elements of the class c
        """
        return self.vd.getElementByClass(c)

    def fit(self, data, epochs=10, batch=1):
        if data is None or len(data) < 1:
            return
        #batch = int(len(data)/10)+1
        self.emb.fit(data, epochs, batch)

    def autofit(self, epoch=10, batch=1, new=False):
        if new:
            self.fit(self.getNewChunks(), epoch, batch)
        else:
            self.fit(self.getAllChunks(), epoch, batch)
        
    def uniqueChunkNumber(self):
        """
        Return the number of unique chunks in the vectorDict.
        """
        return len(self.vd.chunkList)

    def getAllChunks(self):
        return self.vd.getAllChunks()

    def getNewChunks(self):
        return self.vd.getNewChunks()

    def isEmpty(self):
        return self.vd.isEmpty()

    def preComputeVector(self, seq):
        return self.vd.preComputeVector(seq)

    def getChunkFromList(self, index):
        return self.vd.chunkList[index]

    def getActualId(self):
        return self.vd.actualId
    
    def substituteCount(self, new_c, old_c):
        self.vd.classCount[new_c] = self.vd.classCount[old_c]
        self.vd.classCount[old_c] = 0

    
    # GET INFORMATION ABOUT EMBEDDINGS
    #-----------------------------------

    def printClassInfo(self):
        print("\#| CLASS INFO LEVEL ",self.level," |#/")
        # COUNTS
        if len(self.vd.classCount) == 0:
            return
        classCount = self.vd.classCount
        maxCount = max(classCount)
        maxClass = classCount.index(maxCount)
        maxList = self.vd.classList[maxClass]
        minCount = min(classCount)
        minClass = classCount.index(minCount)
        minList = self.vd.classList[minClass]
        meanCount = sum(classCount)/len(classCount)
        print("#== COUNTS ==#")
        print("MAX COUNT: ", maxCount)
        #print("--> For class {} : {}".format(maxClass, maxList))
        print("MIN COUNT: ", minCount)
        #print("--> For class {} : {}".format(minClass, minList))
        print("MEAN COUNT: ", meanCount)

        #DISTANCES
        print("#== DISTANCES ==#")
        meanDist = self.computeMeanDistance()
        print("MEAN DIST: ", meanDist)
        print("____________________________")

    
    
    def csvClassInfo(self):
        res = []
        res.append(self.level)
        try:
            classCount = self.vd.classCount
            maxCount = max(classCount)
        except:
            return res
        meanClassCount = mean(classCount)
        stdClassCount = std(classCount)
        # COUNTS
        res.append(maxCount)
        res.append(meanClassCount)
        res.append(stdClassCount)

        #DISTANCES
        #______________________________________
       
        internal_distance = [] # level distance array (for std)
        for cl in self.vd.classList:          
            for chunks in self.vd.classList[cl]:
                element = np.frombuffer(chunks, dtype='int32')
                cnt = self.vd.classList[cl][chunks]
                base = self.vd.classEmb[cl] if cl in self.vd.classEmb else self.vd.classEmbDeposit[cl]
                element_dist = np.linalg.norm(base - self.emb.get_embedding(element))*cnt
                for _ in range(cnt):
                    internal_distance.append(element_dist)
        
        meanDist = mean(internal_distance)
        stdDist = std(internal_distance)
        res.append(meanDist)
        res.append(stdDist)
        return res

    def computeMeanDistance(self):
        count = 0
        tsum = 0
        for cl in self.vd.classList:
            if len(self.vd.classList[cl].keys()) > 1:
                print("CLASS {} with {}".format(cl, len(self.vd.classList[cl].keys())))
            for chunks in self.vd.classList[cl]:
                element = np.frombuffer(chunks, dtype='int32')
                cnt = self.vd.classList[cl][chunks]
                tsum += np.linalg.norm(self.vd.classEmb[cl] - self.emb.get_embedding(element))*cnt
                count += cnt
        return tsum/count       

    def getBatchEmbedding(self, data):
        res = self.emb.getBatchEmbedding(data)
        return res
    

    def clean(self):
        return self.vd.cleanClass()

    
    def savePartialState(self):
        self.emb.savePartialState()
    
    def loadPartialState(self):
        self.emb.loadPartialState()
    
    def resetPartialState(self):
        self.emb.resetPartialState()
from classes.Embedder import Embedder
from classes.VectorDict import VectorDict

import numpy as np

from time import time

class EmbedSystem():
    def __init__(self, level, dist=10, unit1=1, unit2=1, special_c= -45000, maxlen=0, embedder=None, doMean=False, orthogonal=False, lamb=0.01):
        self.vd = VectorDict(dist, maxlen, level=level, doMean=doMean)
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
            a = time()
            
            if precomputation:
                pre = self.vd.preComputeVector(tokenSeq)
                if pre is not None:
                    #print("YES!")
                    if inlist:
                        self.vd.insertSeq(tokenSeq, tokenSeq, pre, push)
                    return pre
            
            b = time()
            #embseq = tokenSeq#self.emb.get_embedding(tokenSeq)
            c = time()
            res = self.vd.computeClass(tokenSeq, tokenSeq, push, dataLevel=dataLevel) #self.vd.addNewClass(tokenSeq, tokenSeq, push)
            d = time()
            #print("1) {}\n2){}\n3) {}".format(b-a, c-b, d-c))
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


    
    def getValueList(self, c):
        """
        c :- class of elements
        return :- list of elements of the class c
        """
        return self.vd.getElementByClass(c)

    def fit(self, data, epochs=10, batch=10):
        if data is None or len(data) < 1:
            return
        batch = int(len(data)/10)+1
        self.emb.fit(data, epochs, batch)

    def autofit(self, epoch=10, batch=10, new=False):
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

    def deleteClassElement(self, new_c, old_c, chunk):
        """
        c : - class of the chunk
        chunk :- 

        delete chunk from old_c and update the new_c one in ClassList
        """
        self.vd.deleteClassElement(new_c, old_c, chunk)
    
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

    def strClassInfo(self):
        print("  --  LEVEL ", self.level, "  --")
        result = ""
        result += "\#| CLASS INFO LEVEL {} |#/\n".format(self.level)
        # COUNTS
        if len(self.vd.classCount) == 0:
            return ""
        classCount = self.vd.classCount
        result += "Total Classes: {}\n".format(len(classCount))
        maxCount = max(classCount)
        maxClass = classCount.index(maxCount)
        maxList = self.vd.classList[maxClass]
        minCount = min(classCount)
        minClass = classCount.index(minCount)
        minList = self.vd.classList[minClass]
        meanCount = sum(classCount)/len(classCount)
        std = np.std(classCount)
        result += "#== COUNTS ==#\n"
        result += "MAX COUNT: {}\n".format(maxCount)
        #print("--> For class {} : {}".format(maxClass, maxList))
        result += "MIN COUNT: {}\n".format(minCount)
        #print("--> For class {} : {}".format(minClass, minList))
        result += "MEAN COUNT: {}\n".format(meanCount)
        result += "STD: {}\n".format(std)

        #DISTANCES
        result += "#== DISTANCES ==#\n"

        #______________________________________
        c1 = 0 # single class counter
        c2 = 0 # level chunks counter
        s1 = 0 # single class distance accumulator
        s2 = 0 # level distance accumulator
        t = [] # level distance array (fro std)
        for cl in self.vd.classList:
            c1 = 0
            s1 = 0            
            for chunks in self.vd.classList[cl]:
                element = np.frombuffer(chunks, dtype='int32')
                cnt = self.vd.classList[cl][chunks]
                element_dist = np.linalg.norm(self.vd.classEmb[cl] - self.emb.get_embedding(element))*cnt
                s1 += element_dist
                t.append(element_dist)
                c1 += cnt
            c2 += c1
            s2 += s1
            if len(self.vd.classList[cl].keys()) > 1:
                result += "CLASS {} with {}".format(cl, len(self.vd.classList[cl].keys()))
                #print("CLASS {} with {}".format(cl, len(self.vd.classList[cl].keys())))
                result += "\t| mean dist --> {}\n".format(s1/c1)
        meanDist = s2/c2
        stdDist = np.std(t)
        #______________________________________
        result += "MEAN DIST: {} +/- {}\n".format(meanDist, stdDist)
        #result += "STD DIST: {}\n".format(stdDist)
        result += "__________________________\n"
        return result


    def strClassInfo_small(self):
        result = ""
        result += "{})\n".format(self.level)
        # COUNTS
        if len(self.vd.classCount) == 0:
            return ""
        classCount = self.vd.classCount
        result += "Total Classes: {}\n".format(len(classCount))
        maxCount = max(classCount)
        maxClass = classCount.index(maxCount)        
        meanCount = sum(classCount)/len(classCount)
        std = np.std(classCount)
        result += "MEAN COUNT: {} | {}\n".format(meanCount, std)

        #DISTANCES
        #______________________________________
        c1 = 0 # single class counter
        c2 = 0 # level chunks counter
        s1 = 0 # single class distance accumulator
        s2 = 0 # level distance accumulator
        t = [] # level distance array (fro std)
        for cl in self.vd.classList:
            c1 = 0
            s1 = 0            
            for chunks in self.vd.classList[cl]:
                element = np.frombuffer(chunks, dtype='int32')
                cnt = self.vd.classList[cl][chunks]
                element_dist = np.linalg.norm(self.vd.classEmb[cl] - self.emb.get_embedding(element))*cnt
                s1 += element_dist
                t.append(element_dist)
                c1 += cnt
            c2 += c1
            s2 += s1
        meanDist = s2/c2
        stdDist = np.std(t)
        result += "MEAN DIST: {} +/- {}\n\n".format(meanDist, stdDist)

        return result
    
    def csvClassInfo(self):
        res = []
        res.append(self.level)
        try:
            classCount = self.vd.classCount
            maxCount = max(classCount)
            maxClass = classCount.index(maxCount)        
            meanCount = sum(classCount)/len(classCount)
        except:
            return res
        std = np.std(classCount)
        # COUNTS
        res.append(meanCount)
        res.append(std)
        #DISTANCES
        #______________________________________
        c1 = 0 # single class counter
        c2 = 0 # level chunks counter
        s1 = 0 # single class distance accumulator
        s2 = 0 # level distance accumulator
        t = [] # level distance array (fro std)
        for cl in self.vd.classList:
            c1 = 0
            s1 = 0            
            for chunks in self.vd.classList[cl]:
                element = np.frombuffer(chunks, dtype='int32')
                cnt = self.vd.classList[cl][chunks]
                element_dist = np.linalg.norm(self.vd.classEmb[cl] - self.emb.get_embedding(element))*cnt
                s1 += element_dist
                t.append(element_dist)
                c1 += cnt
            c2 += c1
            s2 += s1
        meanDist = s2/c2
        stdDist = np.std(t)
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
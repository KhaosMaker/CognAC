import numpy as np
import pickle
from numpy.linalg import norm
from statistics import mean, stdev

from time import time


class VectorDict():
    def __init__(self, dist, level=0, doMean=False):
        # Class ID : embedding 
        self.classEmb = {}
        # Embedding : Class ID
        self.reverseEmb = {}
        # Class ID : count
        self.classCount = []
        # Class ID : list of chunks of the class
        self.classList = {}
        # ChunkList :- list of all the chunks
        self.chunkList = []

        # Unique list of chunks, with a set for control
        self.uniqueChunkList = []
        self._UCL = set()

        # Reverse vect to class
        self.vectToClass = {}

        self.actualId = 0
        self.dist = dist
        self.doMean = doMean

        # Cunk already fitted in the embedder
        self.alreadyFitted = 0
        self.level = level
    
    def computeClass(self, emb, seq, push=True, dataLevel=False):
        """
        seq :- la sequenza di cui si vuole la classe
        emb :- l'embedding di seq
        return :- the classID of seq.

        NB: If the class is not found, it will be crated
        """
        """
        pre = self.preComputeVector(seq)
        if pre is not None:
            return pre
        """
        c = self.getClass(emb, seq, push, dataLevel)
        if c is not None:
            return c
        else:
            # if it is not found
            newClass = self.addNewClass(emb, seq, push)
            return newClass

    def computeClassSpecial(self, emb, seq, push):
        newClass = self.addNewClass(emb, seq, push)
        return newClass
        
    def getClass(self, emb, seq, push=True, dataLevel=False):
        """
        emb :- l'embedding della sequenza di cui si vuole la classe
        seq :- Sequenza relativa all'embedding
        return :- the classID of seq if it exist, None otherwise.
        """
        if len(self.classEmb.keys()) == 0:
            return None     
        
        temp = np.array(list(self.classEmb.values()))
        temp2 = norm(emb - temp, axis=-1)
        idx = np.argmin(temp2)
        
        if temp2[idx] <= self.dist:
            #print(self.level, ") fromGetClass: ", len(self.reverseEmb))
            # DA CORREGGERE :- perché dovrebbe accadere?
            if temp[idx].tobytes() not in self.reverseEmb:
                return None
            c = self.reverseEmb[temp[idx].tobytes()]
            self.insertSeq(emb, seq, c, push, dataLevel)
            return c
        else:
            return None
    
    def getOnlyClass(self, emb, seq):
        if len(self.classEmb.keys()) == 0:
            return None     
        
        temp = np.array(list(self.classEmb.values()))
        temp2 = norm(emb - temp, axis=-1)
        idx = np.argmin(temp2)
        if temp2[idx] <= self.dist:
            try:
                c = self.reverseEmb[temp[idx].tobytes()]
                return c
            except:
                return None
                for k in self.classEmb:
                    if temp[idx].tobytes() == self.classEmb[k].tobytes():
                        print("<VD>ERROR: ", k)
                        return k
                        
        else:
            return None


    def getNearClass(self, emb):
        try:
            temp = np.array(list(self.classEmb.values()))
            temp2 = norm(emb - temp, axis=-1)
        except:
            return -1
        idx = np.argmin(temp2)
        c = self.reverseEmb[temp[idx].tobytes()]
        return c

    def addNewClass(self, emb, seq, push=True):
        """
        Create a new classID
        """
        nc = len(self.classCount)
        self.classEmb[nc] = np.asarray(emb)
        self.reverseEmb[np.asarray(emb).tobytes()] = nc
        self.classCount.append(1)
        if push:
            self.chunkList.append(seq)
        self.classList[self.actualId] = {}
        key = seq.tobytes()

        if key not in self._UCL:
            self._UCL.add(key)
            self.uniqueChunkList.append(seq)
        self.classList[self.actualId][key] = 1
        self.vectToClass[key] = self.actualId
        self.actualId += 1
        return self.actualId-1


    def insertSeq(self, emb, seq, c, push=True, dataLevel=False):
        """
        Insert a sequence in the vector dictionary model
        """
        # NB SE SI RIMETTE NON FUNZIONA, AGGIUSTARE EMBEDSYSTEM COMPUTECLASS
        if self.doMean:
            temp = self.classEmb[c].tobytes()
            self.classEmb[c] = (self.classEmb[c]*self.classCount[c]+emb)/(self.classCount[c]+1)
            self.reverseEmb[self.classEmb[c].tobytes()] = self.reverseEmb[temp]
            del self.reverseEmb[temp]
            print(self.reverseEmb[self.classEmb[c].tobytes()])

        self.classCount[c] += 1
        if push:
            self.chunkList.append(seq)
        key = seq.tobytes()#np.array_str(seq)

        if key not in self._UCL:
            self._UCL.add(key)
            self.uniqueChunkList.append(seq)

        if key not in self.classList[c]:
            self.classList[c][key] = 1
        else:
            self.classList[c][key] += 1


        self.vectToClass[key] = c


    def compute_distance(self, seq_1, seq_2):
        return norm(seq_1 - seq_2)

    def getElementByClass(self, id):
        """
        return a tuple (sequences, counts)
        """
        result1 = []
        result2 = []
        if not id in self.classList or len(self.classList[id]) == 0:
            #print(self.classList.keys())
            #print(" -- ",id)
            return [np.array([0])], [1]
        for element in self.classList[id]:
            result1.append(np.frombuffer(element, dtype=self.uniqueChunkList[0].dtype))
            result2.append(self.classList[id][element])
        return result1, result2

    def isNear(self, seq_1, seq_2):
        return self.compute_distance(seq_1, seq_2) < self.dist

    def getAllChunks(self):
        return self.chunkList
    
    def getNewChunks(self):
        temp = self.alreadyFitted
        self.alreadyFitted = len(self.chunkList)
        return self.chunkList[temp:self.alreadyFitted]

    def isEmpty(self):
        return self.actualId == 0

    def preComputeVector(self, seq):
        """
        Input :- no array with the chunk
        Output :- class of the sequence or None if not present
        """
        key = seq.tobytes()#np.array_str(seq)
        return self.preComputeVectorByKey(key)

    def preComputeVectorByKey(self, key):
        if key in self.vectToClass:
            return self.vectToClass[key]
        return None


    def deleteClassElement(self, new_c, old_c, chunk):
        strseq = chunk.tobytes()#np.array_str(chunk)
        self.classList[new_c][strseq] = self.classList[old_c][strseq]
        #del self.classList[old_c][strseq]
    
    # SAVE
    def getSave(self):
        res = {}
        res["classEmb"] = self.classEmb
        res["classList"] = self.classList
        res["actualId"] = self.actualId
        res["level"] = self.level
        res["dist"] = self.dist
        return res
    
    def load(self, save, type='int32'):
        self.classEmb = save['classEmb']
        self.classList = save['classList']
        self.actualId = save['actualId']
        self.level = save['level']
        self.dist = save['dist']
        
        
        # Class ID : count
        for c in self.classList:
            self.classCount.append(len(self.classList[c]))
            for ch in self.classList[c]:
                seq = np.frombuffer(ch, dtype='int32')
                if ch not in self._UCL:
                    self._UCL.add(ch)
                    self.uniqueChunkList.append(seq)
                    self.vectToClass[ch] = c


    def reset(self):
        # Class ID : embedding 
        self.classEmb = {}
        self.reverseEmb = {}
        # Class ID : count
        self.classCount = []
        # Class ID : list of chunks of the class
        self.classList = {}
        # ChunkList :- list of all the chunks
        self.chunkList = []

        # Reverse vect to class
        self.vectToClass = {}

        self.actualId = 0

        # Cunk already fitted in the embedder
        self.alreadyFitted = 0


    def cleanClass(self):
        """
        Delete the low used classes. Return the deleted ones.
        """
        res = []
        for c in range(len(self.classCount)):
            if len(self.classEmb) < 500:
                return res
            if self.classCount[c] == 1:
                self.classCount[c] = 0
                #for v in self.classList[c]:
                #    del self.vectToClass[v]
                #del self.classList[c]
                if c in self.classEmb:
                    del self.classEmb[c]
                    #emb = self.classEmb[c].tobytes()
                    #del self.reverseEmb[emb]
                res.append(c)
        return res
    
    def meanChunkLen(self):
        if len(self.uniqueChunkList) < 3:
            return 0,0
        v = [len(chunk) for chunk in self.uniqueChunkList]
        m = mean(v)
        s = stdev(v)
        return m, s


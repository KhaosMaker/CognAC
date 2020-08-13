import numpy as np
import pickle
from numpy.linalg import norm

from time import time


class VectorDict():
    def __init__(self, dist, maxlen=0, level=0, doMean=False):
        # Class ID : embedding 
        self.classEmb = []
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
        self.maxlen = maxlen
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
        if len(self.classEmb) == 0:
            return None     
        
        temp = np.array(self.classEmb)
        temp2 = norm(emb - temp, axis=-1)
        idx = np.argmin(temp2)
        if temp2[idx] <= self.dist:
            self.insertSeq(emb, seq, idx, push, dataLevel)
            return idx
        else:
            return None

    def getNearClass(self, emb):
        temp = np.array(self.classEmb)
        return  np.argmin(norm(emb - temp, axis=-1))

    def addNewClass(self, emb, seq, push=True):
        """
        Create a new classID
        """
        self.classEmb.append(np.asarray(emb))
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
            self.classEmb[c] = (self.classEmb[c]*self.classCount[c]+emb)/(self.classCount[c]+1)
        self.classCount[c] += 1
        if push:
            self.chunkList.append(seq)
        if self.maxlen > 0 and len(self.classList[c]) > self.maxlen:
            self.classList[c].pop(0)
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
        if len(self.classList[id]) == 0:
            print(self.classList.keys())
            print(" -- ",id)
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
        res["maxlen"] = self.maxlen
        return res
    
    def load(self, save):
        self.classEmb = save['classEmb']
        self.classList = save['classList']
        self.actualId = save['actualId']
        self.level = save['level']
        self.dist = save['dist']
        self.maxlen = save['maxlen']

        
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
        self.classEmb = []
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
            if self.classCount[c] == 1:
                self.classCount[c] = 0
                for v in self.classList[c]:
                    del self.vectToClass[v]
                del self.classList[c]
                del self.classEmb[c]
                res.append(c)
        return res

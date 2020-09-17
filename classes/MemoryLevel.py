from math import floor, sqrt
from statistics import stdev, mean
import numpy as np

class MemoryLevel:
    def __init__(self, level):
        self.level = level
        self.memory = []
        self.h = []
        self._H = []
        self.alphabet = set()
        #self.reverseAlphabet = {}
        self.memoryLength = 0

        # TEST
        self.chunkMaxLength = 0
        
        # For each element, it contain the index of the element at level-1 that _close_ the chunk
        # a chunk at index i is the union of the elements at level-1 with index from chunks[i-1] to chunks[i]
        self.chunks = [-1]

        # Usend in the boundary function
        self.meanh_w = 0
        self.window = 100
        self.d = 2
    
    def addToMemory(self, item):
        self.memory.append(item)
        self.alphabet.add(item)
        self.memoryLength += 1


    def addh(self, value):
        self.h.append(value)

    
    def hasToChunk(self):
        """
        CHUNKING METHOD HERE
        fom = 1st order model
        """
        """
        if len(self.h) > 2:
            #print(self.level, ") ", self.h[-1], "vs ", self.h[-2])
            return self.h[-1] > self.h[-2]
        else:
            return False
        """
        if len(self.h) > self.window and self.h[-1] > self.h[-2] :
            std = self.stdInformationContent()       
            if std == 0:
                return False     
            self.meanh_w = self.meanInformationContent()  
            """
            if self.level >= 0 and self.memory[len(self.h)-2] > 0:
                print("{} -> {}".format(self.memory[len(self.h)-2], self.memory[len(self.h)-1]))
                print("{}) {} | {} | {}".format(self.level, self.meanh_w, std, (self.h[-1] - self.meanh_w)/std)) 
                print("{} > {}  || {}".format(self.h[-1], self.h[-2], self.h[-1] > self.h[-2] and (self.d < (self.h[-1] - self.meanh_w)/std)))
                input()
            """
            
            return (self.d < (self.h[-1] - self.meanh_w)/std)
        else:
            return False


    def meanInformationContent(self):
        w = min(self.window, len(self.h))
        """
        tot = sum(self.h[-w:])
        return tot/w
        """
        return mean(self.h[-w:-1])
    
    def stdInformationContent(self):
        w = min(self.window, len(self.h))
        """
        num = sum([(self.h[i] - self.meanh_w) for i in range(len(self.h))])
        return num / w
        """
        return stdev(self.h[-w:-1])

    def chunk(self, lowerLevel, index=None):
        """
        Chunk the <lowerLevel> and create the new chunk, adding it to the
        current level.
        Return the list with the composition of the chunk.
        """
        # Is -2 because: -1 for the normal array length usage, 
        # -1 to take the token before the last one (wich close the chunk) 
        if index is None:
            index = lowerLevel.memoryLength - 2
        self.chunks.append(index)
    
        e = self.chunks[-1]+1
        if len(self.chunks) > 2:
            s = self.chunks[-2]+1
        else:
            s = e
                        
        newChunk = lowerLevel.getChunk(s, e)

        return newChunk
    
    def getChunkLimits(self, index):
        """
        return a couple start:end which are the indexes at level-1 that delimits the chunk in the
        <index> at the actual level
        """
        return self.chunks[index]+1, self.chunks[index+1]

    def getChunk(self, start, end):
        """
        start
        end (not incluse)
        return :- numpy array

        NB: if start == end, return a 1 token chunk
        """
        if start == end:            
            return np.array(self.memory[start-1:end], dtype='int32')
        else:
            return np.array(self.memory[start:end], dtype='int32')

    def chunkToString(self, chunk):
        name = ""
        for item in range(len(chunk)-1):
            name += str(chunk[item])+','
        return name+str(chunk[len(chunk)-1])
    
    def stringToChunk(self, string):
        return string.split(',')

    def printLevelStats(self):
        print("Level: ", self.level)
        print("Size: ", self.memoryLength)
        print("Alphabet size: ", len(self.alphabet))
        # How many time each token appear in the memory. Higher is the better (?)
        if len(self.alphabet) <= 0:
            print("Mean appear: 0%")
        else:
            print("Mean appear: ", floor(self.memoryLength/len(self.alphabet) * 100)/100)

    def printLevel(self, max=10, start=0):
        self.printLevelStats()
        for i in range(max):
            if i+start < len(self.memory):
                print(self.memory[i+start], end=' -> ')
            else:
                break

    def returnMemory(self):
        result = []
        for i in range(self.memoryLength):
            result.append(self.memory[i])
        return result


    def actualItemId(self):
        return self.memory[self.memoryLength-1]
        
    def beforeItemId(self):
        return self.memory[self.memoryLength-2]

    def actualItem(self):
        return self.memory[self.memoryLength-1]
        
    def beforeItem(self):
        return self.memory[max(0,self.memoryLength-2)]
    
    def getItem(self, index):
        """
        Return the token at the <index> in memory.
        NB: return the element as a string, not the ID.
        """
        #print(len(self.memory), "vs", index, " vs ", self.memoryLength)
        return self.memory[index]

    def geth(self):
        return self.h
    
    def getFringeh(self):
        """
        Return the information content of the last element on the fringe.
        If the element does not exist, return None.
        """
        if len(self.h) > 0:
            return self.h[len(self.h)-1]
        else:
            return None
    
    def getLastIndex(self):
        return self.memoryLength-1

    def getChunksLen(self):
        return len(self.chunks)
    
    def getLastChunkClosure(self):
        """
        Return the index at level-1 of the item close the last chunk
        and the index of the last chunk itself
        """
        # NB: the last item is at len-1, while the index is -2 for the
        # shadow chunk at position 0.
        return self.chunks[-1], len(self.chunks)-2
    
    def getPreviousChunkClosure(self, index, fringe):
        """
        index :- index of the token at <level-1>
        return :- (closureIndex, chunkIndex) where:
                    - closureIndex :- the index at <level-1> that close the previous chunk
                    - chunkIndex :- the index of this chunk at <level>
        """
        if fringe:
            # if it is on the fringe
            return self.getLastChunkClosure()
        else:
            # if it is inside the memory (so it is already chunked)
            # The largest index in [chunks] < index
            res = len(self.chunks)-1
            for idx in range(0, len(self.chunks)):
                if self.chunks[idx] >= index:
                    res = idx-1
                    break

            # res-1 as resulting index for the shadow chunk in position 0
            return self.chunks[res], res-1


    def getUnchankedTokens(self, first):
        """
        first :- first element without a chunk in the memory
        return :- list of elements not already chunked. Could be empty.
        """
        return self.getChunk(first, self.memoryLength)
    
    def getLastToken(self):
        """
        return the last token in the memory.
        NB Return the token, not the ID.
        """
        return self.memory[self.memoryLength-1]
    
    def deleteLast(self):
        self.memory = self.memory[:-1]
        self.memoryLength -= 1
        self.h = self.h[:-1]
        self.chunks = self.chunks[:-1]

    """
    def possibleChunks(self, tokenList, previousChunk, fom, es):

        tokenList :- list of tokens that compose part of the chunk
        previousChunk :- the chunk wich precedes the possibles chunks
        fom :- first order model of the chunk's layer
        es :- embedSystem of the chunk's layer

        return :- list of all possible chunks that:
                        - start with [tokenList]
                        - could follow the previousChunk (that appear at time t-1)

        subChunk = np.array(tokenList)
        print("SC: ", subChunk)
        classSubChunk = es.getNearClass(subChunk)
        return es.getValueList(classSubChunk)

        #TODO -> usare la forward distribution che ritorna un dictionary con già le probabilità
        # Row of the transition matrix of the previousChunk
        transitionVector = fom.getTransitionMatrixRow(previousChunk)            
        res = []
        subChunk = np.array(tokenList)
        l = len(tokenList)

        # For each possible future class
        for e in transitionVector:
            print("LEN: ", len(es.getValueList(e)))
            res += [x for x in es.getValueList(e) if np.array_equal(subChunk, x[0:l])]

        return res

        

        # For each element in the transition vector, if P > 0 and start with the right seq of tokens
        # add it to the result (as ID, not as index)
        res = []
        for idx in transitionVector:
            if fom.indexToId[idx].replace(',', '').startswith(subChunk):
                res.append(fom.indexToId[idx])
        return res
    """
from classes.MemoryLevel import MemoryLevel
from classes.CONSTANTS import zeroProbability
import math
import numpy as np
from bisect import bisect_left
from time import time

# TEMP
from random import randint, random

class Memory():
    def __init__(self, levels=1):
        self.levels = levels
        self.memory = []
        for l in range(levels):
            self.memory.append(MemoryLevel(l))

    def resetMemory(self):
        for idx in range(0, self.levels):
            self.memory[idx] = MemoryLevel(idx)
    
    def resetMemoryLevel(self, level):
        self.memory[level] = MemoryLevel(level)

    def hasToChunk(self, level, fom, es, previousIndex = None, token = None):
        """
        Return true if it can be created a new chunk at <level>
        """
        # If previous index and token aren't given, take the fringe token and its
        # previous index at <level>
        l = self.memory[level].memoryLength
        if previousIndex is None:
            previousIndex = l - 2
        
        if token is None:
            token = self.memory[level].getItem(l-1)

        h = self.computeInformationContent(level, fom, previousIndex, token, es)
        self.memory[level].addh(h)
        return self.memory[level].hasToChunk()

    def chunk(self, level, index=None):
        """
        Chunk correctly the memory at <level>-1, adding the new chunk at <level>.
        """
        return self.memory[level].chunk(self.memory[level-1], index)
    
    def chunkToString(self, chunk, level):
        """
        Return the chunk as a string "<element>,<element>,..."
        """
        return self.memory[level].chunkToString(chunk)

    def computeInformationContent(self, level, fom, previousIndex, token, es):
        """
        compute Information Content
        -log(p)
        fom = 1st order model
        """
        
        p = self.computeProbability(level, fom, previousIndex, token, es)
        #print("P: {}".format(p))
        h = -math.log(p, 2)
        return h


    def computeProbability(self, level, fom, previousIndex, token, es):
        """
        Compute the probability P of a given element.
        level :- level of the memory
        fom :- First Order Model
        previousIndex :- index of the element before the token
        token :- element whose probability is calculater
        es :- embed system
        """
        result = 0
        # Transition probability (forward) in the same level
        # P(S2 | S1)
        # S1
        bItem = self.memory[level].getItem(previousIndex)
        #aItem = self.memory[level].getItem(index)
        # S2
        aItem = token
        forwardProbability = fom[level].getProbability(bItem, aItem)
        if forwardProbability < zeroProbability:
            return zeroProbability

        # Downward probability
        # Sum_{all possible C_t}(P(S2 | C_t) * P(C_t | C_{t-1}))
        # METTERE A POSTO! TODO!!!
        _t = time()
        downwardProbability = self.computeDownwardProbability(level, fom, previousIndex, token, es)
        _t = time()-_t
        #print("DwP: ", _t)

        # downwardProbability = 1
        # Final Probability
        # Forward * Downward

        result = forwardProbability*max(downwardProbability, zeroProbability)
        return result

    def computeDownwardProbability(self, level, fom, previousIndex, token, embedSystem):
        """
        Compute the downward probability influence given by the token at <level+1>.
    
        level :- level of the memory
        fom :- first order model
        previousIndex :- index of the element before the token
        token :- element whose probability is calculater
        """

        # if this is the last level there is no downward prob
        if level+1 >= self.levels:
            return 1

        if self.memory[level+1].memoryLength > 0:
            # GET all the possible Chunks that can be generated
            # 1. pick the index at <level> that close the previous chunk at <level+1>
            #   X  ->    Y
            # |   \     |   \
            # a -> b -> c -> d
            # In the ex, if the token is c, we want the index of b. The same if the token
            # is d. If it is b, we want the index of the elem _before_ a.
            """OLD: (only on the fringe) unchunkIndex = self.memory[level+1].getLastChunkClosure()"""
            fringe = self.memory[level].memoryLength-1 <= previousIndex+1
            closureIndex, previousChunk = self.memory[level+1].getPreviousChunkClosure(previousIndex+1, fringe)


            
            # 2. get all the elements from closureIndex+1 to the item before the <token> at <level>
            # in the ex above, if token is "b", this result would be the chunk X = [a,b]
            partialChunk = self.memory[level].getChunk(closureIndex+1, previousIndex+2)


            
            # 3b. Get the class of the partialChunk
            pcClass = embedSystem[level+1].getNearClass(partialChunk)
            if pcClass == -1:
                return zeroProbability

            # 4b
            F = fom[level+1].getProbability(self.memory[level+1].getItem(previousChunk), pcClass)

            """
            st = time()
            # TEST v2     vvvvvvvvvvvvvvvv
            # 3c. Get all classes of chunks that start with it
            
            # Get all unique chunks
            x = embedSystem[level].vd.uniqueChunkList

            # Get the len of the partial chunk with and without the last token.
            ln1 = len(partialChunk)
            ln2 = ln1-1

            # Get the byte repr. of the partial chunk with and without the last token.
            tb1 = partialChunk.tobytes()
            # As byte rep minus last element
            tb2 = tb1[:-8]

            # Create a list with all the chunks that match in the form of keys
            # and count.
            y = []
            simCounter = 0
            # the window is a test 
            for idx in range(len(x)): #range(ax(0, len(x)-22050), len(x)):
                if x[idx].shape[0] >= ln1:
                    brep = x[idx].tobytes()
                    if brep[:(-8*ln1)] == tb1:
                        y.append(brep)
                        simCounter += 1
                    else:
                        if brep[:(-8*ln2)] == tb2:
                            simCounter += 1

            if len(y) == 0:
                return zeroProbability

            P = len(y) / simCounter
            # get all possible classes
            
            pcClasses = map(embedSystem[level].vd.preComputeVectorByKey, y)

            # END TEST V2 ^^^^^^^^^^^^^^^^
            et = time()
           

            # 4c. Fw Pr of pcClass
            F = 0
            for pcClass in pcClasses:
                F += fom[level+1].getProbability(self.memory[level+1].getItem(previousChunk), pcClass)
            
            # 5c. P from above
            F = F*P
            """

            # 5c
            if F > zeroProbability:
                aboveP = self.computeDownwardProbability(level+1, fom, previousChunk, pcClass, embedSystem)
                return max(F*aboveP, zeroProbability)
            else:
                return F            
        else:
            # if the upper layer memory is empty or have only 1 item
            return 1
        

         
    
    def getStartOfChunk(self, level, index):
        """
        Given a chunk, return the index at level-1 of the first element of this chunk

        level :- level of the given chunk
        index :- index of the chunk of interest
        return :- index of the element at level-1 that is the first in the chunk
        """
        # Is index and not index-1 because it need to add +1 
        # for the shadow 0 at the start of chunks

        # Add a +1 at the result because the chunks[index] is the last
        # of the previous chunk, so the next is the start of this chunk.

        # If the index exist (ie it is looking for an already formed chunk)
        # return the chunk start
        # else
        # return the first item at the end of the memory that is out of a chunk
        # EXAMPLE
        #  A
        # |  \
        # a -> b -> c -> d
        #
        # in the ex, return c.
        if index < len(self.memory[level].chunks):
            return self.memory[level].chunks[index]+1
        else:
            return self.memory[level].chunks[self.memory[level].getLastIndex()]+1
    
    def getEndOfChunk(self, level, index):
        """
        Given a chunk, return the index at level-1 of the last element of this chunk

        level :- level of the given chunk
        index :- index of the chunk of interest
        return :- index of the element at level-1 that is the last in the chunk
        """
        # It is [index+1] because the shadow 0 at the start of chunks.

        # If the index exist (ie it is looking for an already formed chunk)
        # return the chunk end
        # else
        # return the last item at the end of the memory that is out of a chunk
        # EXAMPLE
        #  A
        # |  \
        # a -> b -> c -> d
        #
        # in the ex, return d.
        if index < len(self.memory[level].chunks):
            return self.memory[level].chunks[index+1]
        else:
            return self.memory[level-1].getLastIndex()

    def normalize(self, distribution):
        total = sum(distribution.values())
        for k in distribution:
            distribution[k] = distribution[k]/max(total,1)
        return distribution
    
    def getLevel(self, level):
        return self.memory[level]
    
    def beforeItem(self, level):
        return self.memory[level].beforeItem()

    def actualItem(self, level):
        return self.memory[level].actualItem()
    
    def addToMemory(self, level, data):
        self.memory[level].addToMemory(data)

    def getItem(self, level, index):
        """
        return the element at position <index> in the
        memory-level <level>.
        NB: return the element, not the ID.
        """
        return self.memory[level].getItem(index)

    def getFringeh(self, level):
        """
        Return the information content of the last item at <level>
        """
        return self.memory[level].getFringeh()
    
    def getFringeToken(self, level):
        return self.memory[level].getLastToken()

    def addMemoryLevel(self):
        self.levels += 1
        self.memory.append(MemoryLevel(self.levels-1))

    

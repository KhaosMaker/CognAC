import numpy as np

class FOM4:
    def __init__(self, level=0, size=100, kind='f'):
        self.size = size
        self.transitionMatrix = np.zeros((self.size, self.size))
        self.first_idToIndex = {}
        self.first_indexToId = {}
        self.second_idToIndex = {}
        self.second_indexToId = {}
        self.first_actualId = 0
        self.second_actualId = 0
        self.level = level
        self.kind = kind

    def resizeMatrix(self):
        temp = np.zeros((self.size*2, self.size*2))
        temp [:-self.size, :-self.size] = self.transitionMatrix
        self.transitionMatrix = temp
        self.size = self.size*2

    def _update(self, first, second):
        while(first >= self.size or second >= self.size):
            self.resizeMatrix()

        self.transitionMatrix[first][second] += 1

    def update(self, first, second):
        if first not in self.first_idToIndex:
            # create the new entries in the dicts
            # and add the position in the transitionMatrix
            self.first_idToIndex[first] = self.first_actualId
            self.first_indexToId[self.first_actualId] = next
            self.first_actualId += 1
            f = self.first_actualId-1
        else:
            f = self.first_idToIndex[first]
        
        if second not in self.second_idToIndex:
            self.second_idToIndex[second] = self.second_actualId
            self.second_indexToId[self.second_actualId] = second
            self.second_actualId += 1
            s = self.second_actualId-1
        else:
            s = self.second_idToIndex[second]
        
        self._update(f, s)

    def reconstruct(self, fromML, toML=None):
        if toML is None:
            #FORWARD
            for idx in range(fromML.memoryLength):
                self.update(fromML.getItem(max(idx-1,0)), fromML.getItem(idx))
            return
        
        # IF one of the two levels are empty
        if fromML.memoryLength * toML.memoryLength == 0:
            return

        if fromML.memoryLength > toML.memoryLength:
            # UPWARD
            chunksLimits = toML.chunks   
            self.update(fromML.getItem(0), toML.getItem(0))
            for idx in range(1, len(chunksLimits)-1):
                fromToken = fromML.getItem(chunksLimits[idx]+1)
                toToken = toML.getItem(idx)
                self.update(fromToken, toToken)
        else:
            #downward model
            chunksLimits = fromML.chunks 
            for idx in range(1, len(chunksLimits)-1):
                fromToken = fromML.getItem(idx)
                toToken = toML.getItem(chunksLimits[idx+1])
                self.update(fromToken, toToken)

    def getValue(self, prev, next):
        """
        Get counting of apperance of two sequential chunks
        #(next | prev), retrieved from the matrix
        """
        try:
            result = self.transitionMatrix[self.first_idToIndex[prev], self.second_idToIndex[next]]
        except:
            return 0
        return result
    
    def getTotal(self, token):
        row = self.first_idToIndex[token]
        return np.sum(self.transitionMatrix[row, :])

    def _getTotal(self, row):
        return np.sum(self.transitionMatrix[row, :])

    
    def getProbability(self, prev, next):
        """
        Ge the probability P(next | prev) 
        next and prev passed as tokens.
        Return the probability (float)
        """
        res = 0
        try:
            res = self.getValue(prev, next) / self.getTotal(prev)
        except:
            return 0
        return res

    
    def _getProbability(self, prev, next):
        """
        As getProbability() but with input as index.
        """
        return self.getProbability(self.first_indexToId[prev], self.second_indexToId[next])
    
    def getDistribution(self, token):
        """
        Given a token, return two array:
            - the first with the possible choice
            - the second with the probability
        """
        # Possible choice
        pc = []
        # Distribution
        d = []
        if not self.first_idToIndex or not self.second_idToIndex:
            return [], []
        
        if token not in self.first_idToIndex:
            return [], []


        row = self.first_idToIndex[token]
        #print("# LEVEL: ", self.level, " (", self.kind, ") #")
        #print("Token {} => row {}".format(token, row))
        pc = self.transitionMatrix[row, :].nonzero()[0]
        d = self.transitionMatrix[row, pc]/self._getTotal(row)
        #print("distr: {}".format(d))
        #print("Chunks: {}".format(pc))
        pc = [self.second_indexToId[x] for x in pc.tolist()]
        #print("Converted: {}".format(pc))
            
        return pc, d.tolist()

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvv



    
    

        


    


if (__name__ == '__main__'):
    print("First Order Model 4 test")
    f = FOM4(size=3)
    f.transitionMatrix.shape
    f.transitionMatrix[0,0] = 10
    f.transitionMatrix[1,1] = 11
    f.transitionMatrix[2,2] = 12
    f.transitionMatrix[1,2] = 13
    print(f.transitionMatrix)
    f.resizeMatrix()
    print("AFTER RESIZE")
    print(f.transitionMatrix)
    print(f.transitionMatrix.shape)




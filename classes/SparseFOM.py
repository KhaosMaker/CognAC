from classes.FirstOrderModel import FirstOrderModel

from scipy.sparse import coo_matrix
import numpy as np

class SparseFOM(FirstOrderModel):
    
    def __init__(self, level=0):
        self.fromToken = []
        self.toToken = []
        self.data = []
        self.idToIndex = {}
        self.indexToId = {}
        self.actualId = 0
        self.level = level

    def update(self, prev, next):
        if next not in self.idToIndex:
            # create the new entries in the dicts
            # and add the position in the transitionMatrix
            self.idToIndex[next] = self.actualId
            self.indexToId[self.actualId] = next
            self.actualId += 1
    
        if prev not in self.idToIndex:
            # create the new entries in the dicts
            # and add the position in the transitionMatrix
            self.idToIndex[prev] = self.actualId
            self.indexToId[self.actualId] = prev
            self.actualId += 1
        
        self.fromToken.append(self.idToIndex[prev])
        self.toToken.append(self.idToIndex[next])
        self.data.append(1)

    def getValue(self, prev, next):
        """
        Get counting of apperance of two sequential chunks
        #(next | prev), retrieved from the matrix
        """
        return self._getValue(self.idToIndex[prev], self.idToIndex[next])
    
    def _getValue(self, prevI, nextI):
        return self.getTransitionMatrix().toarray()[prevI, nextI]
    
    def getTransitionMatrix(self):
        ft = np.array(self.fromToken)
        tt = np.array(self.toToken)
        return coo_matrix((self.data, (ft, tt)))#, shape=(ft.shape[0], tt.shape[0]))

    def getTotal(self, row):
        return self._getTotal(self.idToIndex[row])
    
    def _getTotal(self, rowI):
        return sum(self.getTransitionMatrix().toarray()[rowI, :])

    
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

        tokenIndex = self.idToIndex[token]
        transitionMatrix = self.getTransitionMatrix()
        for idx in range(len(self.toToken[tokenIndex])):
            if transitionMatrix[tokenIndex, idx] > 0:
                pc.append(self.indexToId[idx])
                d.append(self._getProbability(tokenIndex, idx))
        return pc, d

    def getTransitionMatrixRow(self, token):
        TM = self.getTransitionMatrix()
        if TM.shape[0] <= self.idToIndex[token]:
            return np.array([])
        else:
            return TM.getrow(self.idToIndex[token]).toarray()[0]
    
"""    
    def getPossibleTransitions(self, token):
        ""
        token :- subject element

        return :- list of element with P>0 that can appear after <token>
        ""
        res = []
        row = self.getTransitionMatrix().getrow(self.idToIndex[token])
        for elem in row:
            res.append(self.indexToId[elem])
        return res

"""    
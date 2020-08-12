from classes.FirstOrderModel import FirstOrderModel

class interLevelFOM(FirstOrderModel):
    """
    A first order model that work with 2 different alphabet in the transition matrix
    """
    
    def update(self, prev, next):
        """
        Add the new count to transitionMatrix[prev_index][next_index]
        """
        # If the next do not exist
        if next not in self.idToIndex:
            # create the new entries in the dicts
            # and add the position in the transitionMatrix
            self.idToIndex[next] = self.actualId
            self.indexToId[self.actualId] = next
            self.actualId += 1
            self.transitionMatrix.append([])
        
        if prev not in self.idToIndex:
            self.idToIndex[prev] = self.actualId
            self.indexToId[self.actualId] = prev
            self.actualId += 1
            self.transitionMatrix.append([])

        # Pad with 0s the transition matrix to permit to add the <next> at the right index
        # (only if it was not already inserted)
        if self.idToIndex[next] >= len(self.transitionMatrix[self.idToIndex[prev]]):
            for _ in range(len(self.transitionMatrix[self.idToIndex[prev]])-1, self.idToIndex[next]-1):
                self.transitionMatrix[self.idToIndex[prev]].append(0)
            self.transitionMatrix[self.idToIndex[prev]].append(1)
        else:
            # If the index in transitionMatrix[prev] already exist, increment it by 1
            self.transitionMatrix[self.idToIndex[prev]][self.idToIndex[next]] += 1
            
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
        if not self.idToIndex:
            return [], []
        
        if token not in self.idToIndex:
            return [], []

        tokenIndex = self.idToIndex[token]
        for idx in range(len(self.transitionMatrix[tokenIndex])):
            if self.transitionMatrix[tokenIndex][idx] > 0:
                pc.append(self.indexToId[idx])
                d.append(self._getProbability(tokenIndex, idx))
            
        return pc, d
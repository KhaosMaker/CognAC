from classes.CONSTANTS import zeroProbability

import numpy as np

#temp:
from random import randint, random, choice

class FOM3():

    def __init__(self, level=0, kind='f'):
        self.transitionMatrix = {}
        self.zeroOrder = {}
        #self.idToIndex = {}
        #self.indexToId = {}
        self.actualId = 0
        self.level = level

                

    def update(self, prev, next):
        """
        Add the new count to transitionMatrix[prev_index][next_index]
        """
        if prev not in self.transitionMatrix:
            self.transitionMatrix[prev] = {}
        if next not in self.transitionMatrix[prev]:
            self.transitionMatrix[prev][next] = 0
            self.zeroOrder[next] = 0
        self.transitionMatrix[prev][next] += 1
        self.zeroOrder[next] += 1



    def getValue(self, prev, next):
        """
        Get counting of apperance of two sequential chunks
        #(next | prev), retrieved from the matrix
        """
        if next not in self.transitionMatrix[prev]:
            return 0
        return self.transitionMatrix[prev][next]
    
    def _getValue(self, prevI, nextI):
        pass


    def getTotal(self, row):
        return sum(self.transitionMatrix[row].values())


    def getProbability(self, prev, next):
        """
        Ge the probability P(next | prev)

        next and prev passed as tokens.

        Return the probability (float)
        """

        # If the element do not exist in the matrix
        # i.e. it appear only 1 time, and it is the last token entered
        # (because the item where updated when they are the "prev")

        if prev not in self.transitionMatrix:
            return 1
        
        value = self.getValue(prev, next) 
        tot = self.getTotal(prev) + 1

        # Form Pierce:
        # p(x | y) = p_1(x | y) + gamma(x)*p_0(x)
        return 1/tot * self.getZeroOrderProb(next) + value/tot


    def getZeroOrderProb(self, symbol):
        if symbol not in self.zeroOrder:
            return 1 / sum(self.zeroOrder.values())
        return self.zeroOrder[symbol] / sum(self.zeroOrder.values())


    
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

        if token not in self.transitionMatrix:
            return pc, d
        
        
        total = self.getTotal(token) + 1
        for idx in self.transitionMatrix[token].keys():
            pc.append(idx)
            count = self.getValue(token, idx)
            if count > 0:
                d.append(count/total)
            else:
                d.append(1/total)

        return pc, d

    def getNext(self, prev):
        token = prev
        if token not in self.transitionMatrix:
            return []

        return [k for k in self.transitionMatrix[token]]

    def getRandomToken(self):
        return choice(list(self.transitionMatrix.keys()))

    
    """
    def getTransitionMatrixRow(self, token):
        return list(self.transitionMatrix[self.idToIndex[token]].keys())
        #[self.transitionMatrix[self.idToIndex[token]][v] for v in self.transitionMatrix[self.idToIndex[token]]]
    """
    

    
    def toJson(self):
        """
        Return the FOM as dict
        """
        res = {}
        res["transitionMatrix"] = self.transitionMatrix
        #res["idToIndex"] = self.idToIndex
        #res["indexToId"] = self.indexToId
        res["actualId"] = self.actualId
        res["level"] = self.level
        return res


    def fromJson(self, res):
        """
        Get values from dictionary
        """
        self.transitionMatrix = res["transitionMatrix"]
        for idx in range(len(self.transitionMatrix)):
            self.transitionMatrix[idx] = {int(k): int(self.transitionMatrix[idx][k]) for k in self.transitionMatrix[idx]}
            if len(self.transitionMatrix[idx].keys()) == 0:
                self.transitionMatrix[idx][0] = 1
        #self.idToIndex = res["idToIndex"]
        #for element in self.idToIndex:
        #    self.idToIndex[element] = int(self.idToIndex[element])
        self.indexToId = res["indexToId"]
        self.indexToId = {int(k): self.indexToId[k] for k in self.indexToId}
        self.actualId = res["actualId"]
        self.level = res["level"]

    
    def substituteTransition(self, d, row=True, col=True):
        """
        d :- dictonary old_class : new_class
        Substitute all transition in the matrix A|old to A|new
        row :- if there will be a row substitution
        col :- if there will be a substitution on the column
        """
        if row:
            # For each row, if it is an old, create a new_class row that is equal
            for row in self.transitionMatrix:
                if row in d:
                    self.transitionMatrix[d[row]] = self.transitionMatrix[row]
        
        if col:
            for row in self.transitionMatrix:
                for k in d:
                    if k in self.transitionMatrix[row]:
                        self.transitionMatrix[row][k] -= 1
                        self.transitionMatrix[row][d[k]] = 1

        """if row:
            if f in self.transitionMatrix:
                self.transitionMatrix[t] = self.transitionMatrix[f]
                del self.transitionMatrix[f]
            else:                
                self.transitionMatrix[t] = {}

        if col:
            for k in self.transitionMatrix:
                if f in self.transitionMatrix[k]:
                    self.transitionMatrix[k][t] = self.transitionMatrix[k][f]
                    del self.transitionMatrix[k][f]
        """


    def cleanFOM(self, index):
        """
        Delete <index> (class) from the model
        """
        #del self.idToIndex[token]
        #del self.indexToId[index]
        if index in self.transitionMatrix:
            del self.transitionMatrix[index]
        for element in self.transitionMatrix:
            if index in self.transitionMatrix[element]:
                del self.transitionMatrix[element][index]


"""
    def getPossibleTransitions(self, token):
        ""
        token :- subject element

        return :- list of element with P>0 that can appear after <token>
        ""
        res = []
        for elem in self.transitionMatrix[self.idToIndex[token]]:
            res.append(self.indexToId[elem])
        return res
"""


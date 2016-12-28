import sys, collections, math, random, copy
import numpy as np

import copy_reg
import types
from joblib import Parallel, delayed

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

class AtariFeatureExtractor(object):
    def __init__(self, mode='basic', background=None, parallel=False):
        self.features = set([-1])
        self.mode = mode
        self.background = background
        self.parallel = parallel
        self.numRows = 14
        self.numColumns = 16
        # whichColor[color] returns an array containing all tuples (coordinates) of all tiles containing this color
        self.whichColors = {}

    def extractBasicFeatures(self, state):
        screen = state['screen']
        self.features = [-1] #bias
        self.whichColors = {}

        for i in range(self.numRows):
            for j in range(self.numColumns):
                hasColor = np.zeros(128, bool)
                for r in range(15):
                    for c in range(10):
                        color = screen[(15 * i + r) * 160 + 10 * j + c] / 2
                        hasColor[color] = True
                        if self.background is None or self.background[15 * i + r][10 * j + c] != color:
                            self.features.append(i * 16 * 128 + j * 128 + color)
                for c in range(128):
                    if hasColor[c]:
                        if c not in self.whichColors.keys():
                            self.whichColors[c] = []
                        self.whichColors[c].append((i,j))

        self.features = set(self.features)
        self.whichColors = collections.OrderedDict(sorted(self.whichColors.items()))

    def processBPROSLoop(self, c1):
        numRowOffsets = 2 * self.numRows - 1
        numColumnOffsets = 2 * self.numColumns - 1
        numOffsets = numRowOffsets * numColumnOffsets
        numColorPairs = (1 + 128) * 128 / 2
        numBasicFeatures = 128 * self.numRows * self.numColumns
        bproExistence = np.ones((self.numRows * 2 - 1,self.numColumns * 2 - 1), bool)
        features = []

        for k in range(len(self.whichColors[c1])):
            for h in range(len(self.whichColors[c1])):
                rowDelta = self.whichColors[c1][k][0] - self.whichColors[c1][h][0]
                columnDelta = self.whichColors[c1][k][1] - self.whichColors[c1][h][1]
                newBproFeature = False
                if rowDelta > 0:
                    newBproFeature = True
                elif rowDelta == 0 and columnDelta >= 0:
                    newBproFeature = True
                rowDelta += self.numRows - 1
                columnDelta += self.numColumns - 1
                if newBproFeature and bproExistence[rowDelta][columnDelta]:
                    bproExistence[rowDelta][columnDelta] = False
                    features.append(numBasicFeatures + (128+128-c1+1) * c1/2 * numRowOffsets * numColumnOffsets + rowDelta * numColumnOffsets + columnDelta)

        bproExistence.fill(True)

        for i in range(self.whichColors.keys().index(c1)+1, len(self.whichColors.keys())):
            c2 = self.whichColors.keys()[i]
            for it1 in range(len(self.whichColors[c1])):
                for it2 in range(len(self.whichColors[c2])):
                    rowDelta = self.whichColors[c1][it1][0] - self.whichColors[c2][it2][0]+self.numRows-1
                    columnDelta = self.whichColors[c1][it1][1] - self.whichColors[c2][it2][1]+self.numColumns-1
                    if bproExistence[rowDelta][columnDelta]:
                        bproExistence[rowDelta][columnDelta] = False
                        features.append(numBasicFeatures + (128+128-c1+1) * c1/2 * numRowOffsets * numColumnOffsets + (c2-c1) * numRowOffsets * numColumnOffsets + rowDelta * numColumnOffsets + columnDelta)

            bproExistence.fill(True)

        return features

    def extractBPROSFeatures(self, state):
        self.features = list(self.features)

        if self.parallel:
            ret = Parallel(n_jobs=-1)(delayed(self.processBPROSLoop)(c1) for c1 in self.whichColors.keys())
            for x in ret:
                self.features += x
        else:
            for c1 in self.whichColors.keys():
                self.features += self.processBPROSLoop(c1)
        self.features = set(self.features)

    def extractFeatures(self, state):
        if self.mode == 'basic':
            self.extractBasicFeatures(state)
        elif self.mode == 'bpros':
            self.extractBasicFeatures(state)
            self.extractBPROSFeatures(state)

class FrozenLakeFeatureExtractor(object):
    def __init__(self):
        self.features = []

    def extractFeatures(self, state):
        self.features = [state]

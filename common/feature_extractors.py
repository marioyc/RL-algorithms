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

def processBPROSLoop(numRows, numColumns, c1, whichColors):
    numRowOffsets = 2 * numRows - 1
    numColumnOffsets = 2 * numColumns - 1
    numOffsets = numRowOffsets * numColumnOffsets
    numColorPairs = (1 + 128) * 128 / 2
    numBasicFeatures = 128 * numRows * numColumns
    bproExistence = np.ones((numRows * 2 - 1,numColumns * 2 - 1), bool)
    features = []

    for k in range(len(whichColors[c1])):
        for h in range(len(whichColors[c1])):
            rowDelta = whichColors[c1][k][0] - whichColors[c1][h][0]
            columnDelta = whichColors[c1][k][1] - whichColors[c1][h][1]
            newBproFeature = False
            if rowDelta > 0:
                newBproFeature = True
            elif rowDelta == 0 and columnDelta >= 0:
                newBproFeature = True
            rowDelta += numRows - 1
            columnDelta += numColumns - 1
            if newBproFeature and bproExistence[rowDelta][columnDelta]:
                bproExistence[rowDelta][columnDelta] = False
                features.append(numBasicFeatures + (128+128-c1+1) * c1/2 * numRowOffsets * numColumnOffsets + rowDelta * numColumnOffsets + columnDelta)

    bproExistence.fill(True)

    for i in range(whichColors.keys().index(c1)+1, len(whichColors.keys())):
        c2 = whichColors.keys()[i]
        for it1 in range(len(whichColors[c1])):
            for it2 in range(len(whichColors[c2])):
                rowDelta = whichColors[c1][it1][0] - whichColors[c2][it2][0] + numRows - 1
                columnDelta = whichColors[c1][it1][1] - whichColors[c2][it2][1] + numColumns - 1
                if bproExistence[rowDelta][columnDelta]:
                    bproExistence[rowDelta][columnDelta] = False
                    features.append(numBasicFeatures + (128+128-c1+1) * c1/2 * numRowOffsets * numColumnOffsets + (c2-c1) * numRowOffsets * numColumnOffsets + rowDelta * numColumnOffsets + columnDelta)

        bproExistence.fill(True)

    return features

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

    def extractBPROSFeatures(self, state):
        self.features = list(self.features)

        if self.parallel:
            ret = Parallel(n_jobs=2)(
                    delayed(processBPROSLoop)(self.numRows, self.numColumns, c1, self.whichColors) for c1 in self.whichColors.keys())
            for x in ret:
                self.features += x
        else:
            for c1 in self.whichColors.keys():
                self.features += processBPROSLoop(self.numRows, self.numColumns, c1, self.whichColors)
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

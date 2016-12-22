import sys, collections, math, random, copy
import numpy as np


class AtariFeatureExtractor(object):
    def __init__(self, mode='basic', background=None):
        self.features = set([-1])
        self.mode = mode
        self.background = background
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
                        # is this slowing computation? I don't think so, there aren't so many different colors present in a state
                        if color not in self.whichColors.keys():
                            self.whichColors[color] = []
                        self.whichColors[color].append((i,j))

        self.features = set(self.features)
        self.whichColors = collections.OrderedDict(sorted(self.whichColors.items()))

    def extractBPROSFeatures(self, state):
        self.features = list(self.features)
        #print(self.features)
        screen = state['screen']
        numRowOffsets = 2*self.numRows - 1
        numColumnOffsets = 2*self.numColumns - 1
        numOffsets = numRowOffsets*numColumnOffsets
        numColorPairs = (1+128)*128/2
        numBasicFeatures = 128*self.numRows*self.numColumns
        bproExistence = np.ones((self.numRows*2-1,self.numColumns*2-1), bool)

        for c1 in self.whichColors.keys():
            for k in range(len(self.whichColors[c1])):
                for h in range(len(self.whichColors[c1])):
                    rowDelta = self.whichColors[c1][k][0] - self.whichColors[c1][h][0]
                    columnDelta = self.whichColors[c1][k][1] - self.whichColors[c1][h][1]
                    newBproFeature = False
                    if rowDelta>0:
                        newBproFeature = True
                    elif (rowDelta==0 & columnDelta >=0):
                        newBproFeature = True
                    rowDelta+=self.numRows-1
                    columnDelta+=self.numColumns-1
                    if (newBproFeature & bproExistence[rowDelta][columnDelta]):
                        bproExistence[rowDelta][columnDelta]=False
                        self.features.append(numBasicFeatures+(128+128-c1+1)*c1/2*numRowOffsets*numColumnOffsets+rowDelta*numColumnOffsets+columnDelta)

            bproExistence = np.ones((self.numRows*2-1,self.numColumns*2-1), bool)

            for i in range(self.whichColors.keys().index(c1)+1, len(self.whichColors.keys())):
                c2 = self.whichColors.keys()[i]
                for it1 in range(len(self.whichColors[c1])):
                    for it2 in range(len(self.whichColors[c2])):
                        rowDelta = self.whichColors[c1][it1][0] - self.whichColors[c2][it2][0]+self.numRows-1
                        columnDelta = self.whichColors[c1][it1][1] - self.whichColors[c2][it2][1]+self.numColumns-1
                        if bproExistence[rowDelta][columnDelta]:
                            bproExistence[rowDelta][columnDelta]=False
                            self.features.append(numBasicFeatures+(128+128-c1+1)*c1/2*numRowOffsets*numColumnOffsets+(c2-c1)*numRowOffsets*numColumnOffsets+rowDelta*numColumnOffsets+columnDelta)

                bproExistence = np.ones((self.numRows*2-1,self.numColumns*2-1), bool)

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

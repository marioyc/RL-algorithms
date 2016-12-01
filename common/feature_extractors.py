#import time
#import numpy as np
#import itertools
from collections import Counter

FEATURES_DIR = "features"

class BasicFeatureExtractor(object):
    def __init__(self):
        self.features = Counter()
        #self.count = 0
        #self.sumt = 0

    def calcFeatures(self, state, action):
        #start = time.time()
        #self.count += 1
        screen = state['screen']
        self.features = Counter()

        for i in range(16):
            for j in range(14):
                for r in range(10):
                    for c in range(15):
                        self.features[i * 14 * 128 + j * 128 + screen[(10 * i + r) * 210 + 15 * j + c]] = 1

        #end = time.time()
        #self.sumt += end - start

        #if self.count == 1000:
            #print "avg time = ", self.sumt / 1000, "s, total =", self.sumt / 60, "m"
            #self.count = 0
            #self.sumt = 0

    def __call__(self):
        return self.features.iteritems()

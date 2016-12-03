class BasicFeatureExtractor(object):
    def __init__(self):
        self.features = set()

    def extractFeatures(self, state):
        screen = state['screen']
        self.features = []

        for i in range(16):
            for j in range(14):
                for r in range(10):
                    for c in range(15):
                        self.features.append(i * 14 * 128 + j * 128 + screen[(10 * i + r) * 210 + 15 * j + c])
        self.features = set(self.features)

class FrozenLakeFeatureExtractor(object):
    def __init__(self):
        self.features = []

    def extractFeatures(self, state):
        self.features = [state]

class BasicFeatureExtractor(object):
    def __init__(self, background=None):
        self.features = set()
        self.background = background

    def extractFeatures(self, state):
        screen = state['screen']
        self.features = []

        for i in range(14):#for i in range(16):
            for j in range(16):#for j in range(14):
                for r in range(15):#for r in range(10):
                    for c in range(10):#for c in range(15):
                        color = screen[(15 * i + r) * 160 + 10 * j + c]
                        if self.background is None or self.background[15 * i + r][10 * j + c] != color:
                            self.features.append(i * 16 * 128 + j * 128 + color)
        self.features = set(self.features)

class FrozenLakeFeatureExtractor(object):
    def __init__(self):
        self.features = []

    def extractFeatures(self, state):
        self.features = [state]

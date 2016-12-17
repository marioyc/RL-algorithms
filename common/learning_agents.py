import sys, collections, math, random, copy
import numpy as np

from eligibility_traces import EligibilityTraces

class RLAlgorithm(object):
    """
    :description: abstract class defining the interface of a RL algorithm
    """

    def getAction(self, state):
        raise NotImplementedError("Override me")

    def incorporateFeedback(self, state, action, reward, newState):
        raise NotImplementedError("Override me")

class ValueLearningAlgorithm(RLAlgorithm):
    """
    :description: base class for RL algorithms that approximate the value function.
    """
    def __init__(self, actions, discount, featureExtractor, explorationProb, stepSize):
        """
        :type: actions: list
        :param actions: possible actions to take

        :type discount: float
        :param discount: the discount factor

        :type featureExtractor: callable returning dictionary
        :param featureExtractor: returns the features extracted from a state

        :type explorationProb: float
        :param explorationProb: probability of taking a random action

        :type stepSize: float
        :param stepSize: learning rate
        """
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.stepSize = stepSize

        self.freq_actions = collections.Counter()

    def getQ(self, state, action):
        """
        :description: returns the Q value associated with this state-action pair

        :type state: dictionary
        :param state: the state of the game

        :type action: int
        :param action: the action for which to retrieve the Q-value
        """
        score = 0
        for f in self.featureExtractor.features:
            score += self.weights[(f, action)]
        assert(score < 1e7)
        return score

    def getAction(self, state):
        """
        :description: returns an action accoridng to epsilon-greedy policy

        :type state: dictionary
        :param state: the state of the game
        """
        if random.random() < self.explorationProb:
            chosenAction = random.choice(self.actions)
        else:
            Qvalues = np.array([self.getQ(state, action) for action in self.actions])
            chosenAction = self.actions[np.argmax(Qvalues)]
        self.freq_actions[chosenAction] += 1
        return chosenAction

class SARSALambdaLearningAlgorithm(ValueLearningAlgorithm):
    """
    :description: Class implementing the SARSA Lambda algorithm. This
        class is equivalent to the SARSALearningAlgorithm class when
        self.lambda is set to 0; however, we keep it separate here
        because it imposes an overhead of tracking eligibility
        traces and because it is nice to see the difference between
        the two clearly.
    """
    def __init__(self, actions, discount, featureExtractor, explorationProb, stepSize, threshold, decay):
        """
        :note: please see parent class for params not described here
        """
        super(SARSALambdaLearningAlgorithm, self).__init__(actions, discount, featureExtractor,
                    explorationProb, stepSize)
        self.threshold = threshold
        self.decay = decay
        self.eligibility_traces = EligibilityTraces(threshold, decay)
        self.name = "SARSALambda"
        self.maxFeatVectorNorm = 1
        self.firstReward = 0
        self.sawFirst = False

    def resetTraces(self):
        self.eligibility_traces = EligibilityTraces(self.threshold, self.decay)

    def incorporateFeedback(self, state, action, reward, newState):
        """
        :description: performs a SARSA update

        :type state: dictionary
        :param state: the state of the game

        :type action: int
        :param action: the action for which to retrieve the Q-value

        :type reward: float
        :param reward: reward associated with being in newState

        :type newState: dictionary
        :param newState: the new state of the game

        :type rval: int or None
        :param rval: if rval returned, then this is the next action taken
        """
        self.featureExtractor.extractFeatures(state)
        prediction = self.getQ(state, action)
        self.eligibility_traces.update_all()

        if reward != 0 and not self.sawFirst:
            self.sawFirst = True
            self.firstReward = float(reward)
        if self.sawFirst:
            reward /= self.firstReward

        target = reward
        newAction = None

        for f in self.featureExtractor.features:
            self.eligibility_traces[(f, action)] = 1

        if newState != None:
            # SARSA differs from Q-learning in that it does not take the max
            # over actions, but instead selects the action using it's policy
            # and in that it returns the action selected
            # so that the main training loop may use that in the next iteration
            newAction = self.getAction(newState)
            target += self.discount * self.getQ(newState, newAction)

        if len(self.featureExtractor.features) > self.maxFeatVectorNorm:
            self.maxFeatVectorNorm = len(self.featureExtractor.features)

        update = self.stepSize / self.maxFeatVectorNorm * (prediction - target)
        for f, e in self.eligibility_traces.iteritems():
            self.weights[f] -= update * e
        #print 'reward = {}, prediction = {}, target = {}'.format(reward, prediction, target)
        return newAction

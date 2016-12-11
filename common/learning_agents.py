import sys, collections, math, random, copy
import numpy as np

from eligibility_traces import EligibilityTraces

MAX_FEATURE_WEIGHT_VALUE = 1000

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
        self.numIters = 1
        self.stepSize = stepSize
        self.cur_random_action = self.actions[0]

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
        return score

    def getAction(self, state):
        """
        :description: returns an action accoridng to epsilon-greedy policy

        :type state: dictionary
        :param state: the state of the game
        """
        self.numIters += 1


        if random.random() < self.explorationProb:
            self.cur_random_action = random.choice(self.actions)
            return self.cur_random_action
        else:
            Qvalues = np.array([self.getQ(state, action) for action in self.actions])
            maxAction = self.actions[np.argmax(Qvalues)]
            self.freq_actions[maxAction] += 1
        return maxAction

    def getStepSize(self):
        """
        :description: return the step size
        """
        return self.stepSize

class SARSALearningAlgorithm(ValueLearningAlgorithm):
    """
    :description: Class implementing the SARSA algorithm
    """
    def __init__(self, actions, discount, featureExtractor, explorationProb, stepSize):
        """
        :note: please see parent class for params not described here
        """
        super(SARSALearningAlgorithm, self).__init__(actions, discount, featureExtractor,
                    explorationProb, stepSize)
        self.name = "SARSA"

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
        stepSize = self.stepSize
        prediction = self.getQ(state, action)
        target = reward
        newAction = None
        if newState != None:
            # SARSA differs from Q-learning in that it does not take the max
            # over actions, but instead selects the action using it's policy
            # and in that it returns the action selected
            # so that the main training loop may use that in the next iteration
            newAction = self.getAction(newState)
            target += self.discount * self.getQ(newState, newAction)

        update = stepSize * (prediction - target)
        update = np.clip(update, -self.maxGradient, self.maxGradient)
        for f, v in self.featureExtractor(state, action):
            self.weights[f] = self.weights[f] - update * v
            assert(self.weights[f] < MAX_FEATURE_WEIGHT_VALUE)
        # return newAction to denote that this is an on-policy algorithm
        return newAction

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
            self.sawFirst = False
            self.firstReward = reward
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
            #assert(abs(self.weights[f]) < MAX_FEATURE_WEIGHT_VALUE)
        #print 'reward = {}, prediction = {}, target = {}'.format(reward, prediction, target)
        weights = [v for k,v in self.weights.iteritems()]
        #min_feat_weight = min(weights)
        max_feat_weight = max(weights)
        #avg_feat_weight = np.mean(weights)
        #print 'min = {:.4f}, max = {:.4f}, avg = {:.4f}'.format(min_feat_weight, max_feat_weight, avg_feat_weight)
        assert(max_feat_weight < 1e5)


        return newAction

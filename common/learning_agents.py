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
    def __init__(self, actions, featureExtractor, discount, explorationProb, stepSize):
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
        self.featureExtractor = featureExtractor
        self.discount = discount
        self.explorationProb = explorationProb
        self.stepSize = stepSize
        self.weights = collections.Counter()

        self.freq_actions = collections.Counter()

    def getQ(self, action):
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

    def getAction(self):
        """
        :description: returns an action accoridng to epsilon-greedy policy
        """
        if random.random() < self.explorationProb:
            chosenAction = random.choice(self.actions)
        else:
            Qvalues = np.array([self.getQ(action) for action in self.actions])
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
    def __init__(self, actions, featureExtractor, discount, explorationProb, stepSize, decay, threshold):
        super(SARSALambdaLearningAlgorithm, self).__init__(actions, featureExtractor, discount,
                    explorationProb, stepSize)
        self.threshold = threshold
        self.decay = decay
        self.eligibility_traces = EligibilityTraces(threshold, decay)
        self.name = "SARSALambda"
        self.maxFeatVectorNorm = 1
        self.firstReward = 0
        self.sawFirst = False

    def startEpisode(self, state):
        self.resetTraces()
        self.featureExtractor.extractFeatures(state)

    def resetTraces(self):
        self.eligibility_traces = EligibilityTraces(self.threshold, self.decay)

    def incorporateFeedback(self, state, action, reward, newState, prediction=None, target=None):
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
        self.eligibility_traces.update_all()
        for f in self.featureExtractor.features:
            self.eligibility_traces[(f, action)] = 1

        if reward != 0 and not self.sawFirst:
            self.sawFirst = True
            self.firstReward = float(reward)
        if self.sawFirst:
            reward /= self.firstReward

        if prediction is None:
            prediction = self.getQ(action)

        newAction = None

        if target is None:
            target = reward
            if newState != None:
                # extract features of new state
                self.featureExtractor.extractFeatures(newState)
                # SARSA differs from Q-learning in that it does not take the max
                # over actions, but instead selects the action using it's policy
                # and in that it returns the action selected
                # so that the main training loop may use that in the next iteration
                newAction = self.getAction()
                target += self.discount * self.getQ(newAction)

        if len(self.featureExtractor.features) > self.maxFeatVectorNorm:
            self.maxFeatVectorNorm = len(self.featureExtractor.features)

        update = self.stepSize / self.maxFeatVectorNorm * (prediction - target)
        for f, e in self.eligibility_traces.iteritems():
            self.weights[f] -= update * e

        return newAction

class DoubleSARSALambdaLearningAlgorithm(RLAlgorithm):
    def __init__(self, actions, featureExtractor, discount, explorationProb, stepSize, decay, threshold):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.agent_A = SARSALambdaLearningAlgorithm(actions, featureExtractor, discount, explorationProb, stepSize, decay, threshold)
        self.agent_B = SARSALambdaLearningAlgorithm(actions, featureExtractor, discount, explorationProb, stepSize, decay, threshold)
        self.name = "DoubleSARSALambda"

    def startEpisode(self, state):
        self.agent_A.startEpisode(state)
        self.agent_B.startEpisode(state)

    def getAction(self, state):
        if random.random() < self.explorationProb:
            chosenAction = random.choice(self.actions)
        else:
            Qvalues = np.array([self.agent_A.getQ(action) + self.agent_B.getQ(action) for action in self.actions])
            chosenAction = self.actions[np.argmax(Qvalues)]
        return chosenAction

    def incorporateFeedback(self, state, action, reward, newState):
        target = reward
        newAction = None
        if random.random() < 0.5:
            prediction = self.agent_A.getQ(action)
            if newState != None:
                # featureExtractor is the same object for both agents
                # no need to extract features for the other agent
                self.agent_A.featureExtractor.extractFeatures(newState)
                newAction = self.agent_A.getAction()
                target += self.discount * self.agent_B.getQ(newAction)
            self.agent_A.incorporateFeedback(state, action, reward, newState, prediction=prediction, target=target)
        else:
            prediction = self.agent_B.getQ(action)
            if newState != None:
                self.agent_B.featureExtractor.extractFeatures(newState)
                newAction = self.agent_B.getAction()
                target = self.discount * self.agent_A.getQ(newAction)
            self.agent_B.incorporateFeedback(state, action, reward, newState, prediction=prediction, target=target)
        return newAction

import learning_agents

#  default values of parameters common to all agents
ACTIONS = [0,1,3,4]
DISCOUNT = .99#.993
EXPLORATION_PROBABILITY = 0.05#0.01#1
STEP_SIZE = 0.01#0.5 #.001

def build_sarsa_agent(featureExtractor):
    print 'building sarsa agent...'
    return learning_agents.SARSALearningAlgorithm(
                actions=ACTIONS,
                discount=DISCOUNT,
                featureExtractor=featureExtractor,
                explorationProb=EXPLORATION_PROBABILITY,
                stepSize=STEP_SIZE)

# default values of sarsa lambda parameters
THRESHOLD = .1
DECAY = 0.9 * DISCOUNT #.98

def build_sarsa_lambda_agent(featureExtractor):
    print 'building sarsa lambda agent...'
    return learning_agents.SARSALambdaLearningAlgorithm(
                actions=ACTIONS,
                discount=DISCOUNT,
                featureExtractor=featureExtractor,
                explorationProb=EXPLORATION_PROBABILITY,
                stepSize=STEP_SIZE,
                threshold=THRESHOLD,
                decay=DECAY)

import learning_agents

#  default values of parameters common to all agents
DISCOUNT = .99#.993

def build_sarsa_agent(actions, featureExtractor, explorationProb=0.1, stepSize=0.1):
    print 'building sarsa agent...'
    return learning_agents.SARSALearningAlgorithm(
                actions=actions,
                discount=DISCOUNT,
                featureExtractor=featureExtractor,
                explorationProb=explorationProb,
                stepSize=stepSize)

# default values of sarsa lambda parameters
THRESHOLD = .1
DECAY = 0.9 * DISCOUNT #.98

def build_sarsa_lambda_agent(actions, featureExtractor, explorationProb=0.1, stepSize=0.1):
    print 'building sarsa lambda agent...'
    return learning_agents.SARSALambdaLearningAlgorithm(
                actions=actions,
                discount=DISCOUNT,
                featureExtractor=featureExtractor,
                explorationProb=explorationProb,
                stepSize=stepSize,
                threshold=THRESHOLD,
                decay=DECAY)

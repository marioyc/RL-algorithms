from collections import Counter
import random
random.seed(42)

import matplotlib.pyplot as plt
import numpy as np
import gym

import common.feature_extractors as feature_extractors
import common.learning_agents as learning_agents

env = gym.make('FrozenLake-v0')
n = env.observation_space.n

# training parameters
FEATURE_EXTRACTOR = feature_extractors.FrozenLakeFeatureExtractor()
agent = learning_agents.SARSALambdaLearningAlgorithm(
            actions=range(env.action_space.n),
            featureExtractor=FEATURE_EXTRACTOR,
            discount=0.999,
            explorationProb=0.15,
            stepSize=0.2,
            decay=0.9 * 0.999,
            threshold=0.01)
NUM_EPISODES = 50000

# training options
PRINT_TRAINING_INFO_PERIOD = 1000
NUM_EPISODES_AVERAGE_REWARD_OVER = 100

rewards = []
mean_rewards = []
min_weight = []
max_weight = []
avg_weight = []

for episode in range(NUM_EPISODES):
    agent.resetTraces()
    state = env.reset()
    action = 0
    new_action = None
    total_reward = 0
    done = False

    while not done:#for t in range(5000):
        if new_action is None:
            action = agent.getAction(state)
        else:
            action = new_action
        new_state, reward, done, info = env.step(action)
        if done:
            new_state = None
        total_reward += reward
        new_action = agent.incorporateFeedback(state, action, reward, new_state)
        state = new_state

    rewards.append(total_reward)
    mean_reward = np.mean(rewards[-NUM_EPISODES_AVERAGE_REWARD_OVER:])
    mean_rewards.append(mean_reward)

    if episode % PRINT_TRAINING_INFO_PERIOD == 0:
        print '\n############################'
        print '### training information ###'
        print 'Episode: {}'.format(episode)
        print 'Average reward: {}'.format(np.mean(rewards))
        print 'Last 100: {}'.format(mean_reward)
        print 'Exploration probability: {}'.format(agent.explorationProb)
        print 'size of weights dict: {}'.format(len(agent.weights))
        weights = [v for k,v in agent.weights.iteritems()]
        min_feat_weight = min(weights)
        min_weight.append(min_feat_weight)
        max_feat_weight = max(weights)
        max_weight.append(max_feat_weight)
        avg_feat_weight = np.mean(weights)
        avg_weight.append(avg_feat_weight)
        print 'min feature weight: {}'.format(min_feat_weight)
        print 'max feature weight: {}'.format(max_feat_weight)
        print 'average feature weight: {}'.format(avg_feat_weight)
        print '############################\n'

print agent.freq_actions

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(mean_rewards[NUM_EPISODES_AVERAGE_REWARD_OVER - 1:])

plt.subplot(2,1,2)
x_axis = range(0,NUM_EPISODES,PRINT_TRAINING_INFO_PERIOD)
plt.plot(x_axis, min_weight, x_axis, avg_weight, x_axis, max_weight)
plt.show()

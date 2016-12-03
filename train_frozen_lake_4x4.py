from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import gym

import common.feature_extractors as feature_extractors
import common.build_agent as build_agent

env = gym.make('FrozenLake-v0')
n = env.observation_space.n

# training parameters
FEATURE_EXTRACTOR = feature_extractors.FrozenLakeFeatureExtractor()
agent = build_agent.build_sarsa_lambda_agent(
            range(env.action_space.n),
            FEATURE_EXTRACTOR,
            explorationProb=0.4,
            stepSize=0.5)
NUM_EPISODES = 100000

# training options
PRINT_TRAINING_INFO_PERIOD = 1000
NUM_EPISODES_AVERAGE_REWARD_OVER = 100

rewards = []
mean_rewards = []

for episode in range(NUM_EPISODES):
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
        total_reward += reward
        new_action = agent.incorporateFeedback(state, action, reward, new_state)
        state = new_state

        if done:
            #if reward == 1:
            #    print "Episode {} won".format(episode)
            break

    rewards.append(total_reward)
    mean_reward = np.mean(rewards[-NUM_EPISODES_AVERAGE_REWARD_OVER:])
    mean_rewards.append(mean_reward)
    #print "Episode {} finished, reward = {}".format(episode, reward)

    if episode % PRINT_TRAINING_INFO_PERIOD == 0:
        print '\n############################'
        print '### training information ###'
        print 'Episode: {}'.format(episode)
        print 'Average reward: {}'.format(np.mean(rewards))
        print 'Last 100: {}".format(mean_reward)
        print 'Exploration probability: {}'.format(agent.explorationProb)
        print 'size of weights dict: {}'.format(len(agent.weights))
        weights = [v for k,v in agent.weights.iteritems()]
        min_feat_weight = min(weights)
        max_feat_weight = max(weights)
        avg_feat_weight = np.mean(weights)
        print 'min feature weight: {}'.format(min_feat_weight)
        print 'max feature weight: {}'.format(max_feat_weight)
        print 'average feature weight: {}'.format(avg_feat_weight)
        print '############################\n'

print agent.freq_actions
plt.plot(mean_rewards)
plt.show()

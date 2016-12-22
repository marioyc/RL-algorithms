"""
:description: train an agent to play a game
"""
import os
import sys
import time
import numpy as np
import json
import random
random.seed(42)

# atari learning environment imports
from ale_python_interface import ALEInterface

# custom imports
import common.feature_extractors as feature_extractors
import common.file_utils as file_utils
import common.learning_agents as learning_agents

# load config
f = open('config.json')
config = json.load(f)['basic']

# training parameters
GAME = 'space_invaders'
FEATURE_EXTRACTOR = feature_extractors.AtariFeatureExtractor(background=file_utils.load_background(GAME))

# training options
LOAD_WEIGHTS = False
LOAD_WEIGHTS_FILENAME = ''
PRINT_TRAINING_INFO_PERIOD = 10
NUM_EPISODES_AVERAGE_REWARD_OVER = 50
RECORD_BEST = True

# import opencv if necessary
if RECORD_BEST:
    import cv2
    major = cv2.__version__.split(".")[0]

def train_agent(gamepath, agent):
    """
    :description: trains an agent to play a game

    :type gamepath: string
    :param gamepath: path to the binary of the game to be played

    :type agent: subclass RLAlgorithm
    :param agent: the algorithm/agent that learns to play the game
    """

    # load the ale interface to interact with
    ale = ALEInterface()
    ale.setInt('random_seed', 42)
    ale.setFloat("repeat_action_probability", 0.00);
    ale.setInt("frame_skip", config['frame_skip'])
    ale.setBool("color_averaging", True);
    ale.loadROM(gamepath)

    screen_dims = ale.getScreenDims();
    assert(screen_dims[0] == 160 and screen_dims[1] == 210)

    # statistics
    rewards = []
    avgs_rewards_all = []
    avgs_rewards_partial = []
    dict_sizes = []
    mins_feat_weights = []
    maxs_feat_weights = []
    avgs_feat_weights = []
    num_frames = []
    avgs_frames_all = []
    avgs_frames_partial = []

    best_reward = 0

    print('starting training...')
    for episode in xrange(config['train_episodes']):
        agent.resetTraces()
        action = 0
        newAction = None
        reward = 0
        total_reward = 0
        counter = 0

        screen = np.zeros((160 * 210), dtype=np.int8)
        state = {"screen" : screen}
        video_frames = []

        start = time.time()

        while not ale.game_over():
            # if newAction is None then we are training an off-policy algorithm
            # otherwise, we are training an on policy algorithm
            if newAction is None:
                action = agent.getAction(state)
            else:
                action = newAction

            reward = ale.act(action)
            total_reward += reward

            if not ale.game_over():
                new_screen = ale.getScreen()
                if RECORD_BEST:
                    video_frames.append(ale.getScreenRGB())
                new_state = {"screen": new_screen}
            else:
                new_state = None

            newAction = agent.incorporateFeedback(state, action, reward, new_state)
            state = new_state
            counter += 1

        end = time.time()

        print 'episode: {}, score: {}, number of frames: {}, time: {:.4f}m'.format(episode, total_reward, counter, (end - start) / 60)

        if total_reward > best_reward:
            best_reward = total_reward
            print 'Best reward: {}'.format(total_reward)

            if RECORD_BEST:
                video_filename = 'video/{}-{}-{}.avi'.format(GAME, agent.name, episode)
                if major == '2':
                    fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
                else:
                    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
                video = cv2.VideoWriter(video_filename, fourcc, 24, screen_dims)

                for frame in video_frames:
                    video.write(frame)
                video.release()
                file_utils.save_weights(agent.weights, filename='{}-{}-{}'.format(GAME, agent.name, episode))

        # add statistics of current episode
        rewards.append(total_reward)
        avgs_rewards_all.append(np.mean(rewards))
        avgs_rewards_partial.append(np.mean(rewards[-NUM_EPISODES_AVERAGE_REWARD_OVER:]))
        dict_sizes.append(len(agent.weights))
        weights = [v for k,v in agent.weights.iteritems()]
        mins_feat_weights.append(min(weights))
        maxs_feat_weights.append(max(weights))
        avgs_feat_weights.append(np.mean(weights))
        num_frames.append(counter)
        avgs_frames_all.append(np.mean(num_frames))
        avgs_frames_partial.append(np.mean(num_frames[-NUM_EPISODES_AVERAGE_REWARD_OVER:]))
        # save statistics
        file_utils.save_stats(rewards, avgs_rewards_all, avgs_rewards_partial,
                    dict_sizes, mins_feat_weights, maxs_feat_weights, avgs_feat_weights,
                    num_frames, avgs_frames_all, avgs_frames_partial,
                    filename='{}-{}'.format(GAME, agent.name))

        ale.reset_game()

if __name__ == '__main__':
    game = GAME + '.bin'
    gamepath = os.path.join('..', 'roms', game)
    ale = ALEInterface()
    ale.loadROM(gamepath)

    agent = learning_agents.SARSALambdaLearningAlgorithm(
                actions=ale.getMinimalActionSet(),
                featureExtractor=FEATURE_EXTRACTOR,
                discount=config['gamma'],
                explorationProb=config['exploration_probability'],
                stepSize=config['step'],
                decay=config['lambda'] * config['gamma'],
                threshold=config['elegibility_traces_threshold'])

    if LOAD_WEIGHTS:
        agent.weights = file_utils.load_weights(WEIGHTS_FILENAME)

    train_agent(gamepath, agent)

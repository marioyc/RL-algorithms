import json
import logging
import numpy as np
import os
import random
import time

from tqdm import tqdm

# atari learning environment imports
from ale_python_interface import ALEInterface

# custom imports
import common.feature_extractors as feature_extractors
import common.file_utils as file_utils
import common.learning_agents as learning_agents

# load config file
FEATURES = 'bpros'
f = open('config.json')
config = json.load(f)[FEATURES]

# training parameters
GAME = 'space_invaders'
LOAD_WEIGHTS = False
LOAD_WEIGHTS_FILENAME = ''
NUM_EPISODES_AVERAGE_REWARD_OVER = 50
RECORD_BEST = True

random.seed(42)

def train_agent(ale, agent):
    """
    :description: trains an agent to play a game
    :type gamepath: string
    :param gamepath: path to the binary of the game to be played
    :type agent: subclass RLAlgorithm
    :param agent: the algorithm/agent that learns to play the game
    """

    screen_dims = ale.getScreenDims()
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

    # flag for first non-zero reward
    sawFirst = False

    logging.info('Starting training')
    for episode in tqdm(range(config['train_episodes'])):
        action = 0
        newAction = None
        reward = 0
        total_reward = 0
        counter = 0
        newAction = random.choice(agent.actions)

        screen = np.zeros((160 * 210), dtype=np.int8)
        state = {"screen" : screen}
        agent.startEpisode(state)

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

            if reward != 0 and not sawFirst:
                sawFirst = True
                firstReward = float(reward)
            if sawFirst:
                scaledReward = reward / firstReward
            else:
                scaledReward = reward

            if not ale.game_over():
                new_screen = ale.getScreen()
                if RECORD_BEST:
                    video_frames.append(ale.getScreenRGB())
                new_state = {"screen": new_screen}
            else:
                new_state = None

            newAction = agent.incorporateFeedback(state, action, scaledReward, new_state)
            state = new_state
            counter += 1

        end = time.time()

        logging.info('episode: %d, score: %d, number of frames: %d, time: %.4fm', episode, total_reward, counter, (end - start) / 60)
        filename_prefix = "{}-{}-{}".format(GAME, agent.name, FEATURES)
        filename = "{}-{}".format(filename_prefix,  episode)

        if total_reward > best_reward:
            best_reward = total_reward
            logging.info('Best reward: %d', total_reward)

            if RECORD_BEST:
                #file_utils.save_videos(video_frames, screen_dims, filename)
                file_utils.save_weights(agent.weights, filename)

        # update and plot statistics of current episode
        rewards.append(total_reward)
        avgs_rewards_all.append(np.mean(rewards))
        avgs_rewards_partial.append(np.mean(rewards[-NUM_EPISODES_AVERAGE_REWARD_OVER:]))

        num_frames.append(counter)
        avgs_frames_all.append(np.mean(num_frames))
        avgs_frames_partial.append(np.mean(num_frames[-NUM_EPISODES_AVERAGE_REWARD_OVER:]))

        dict_sizes.append(len(agent.weights))

        weights = [v for k,v in agent.weights.iteritems()]
        mins_feat_weights.append(min(weights))
        maxs_feat_weights.append(max(weights))
        avgs_feat_weights.append(np.mean(weights))

        file_utils.plot_stats(avgs_rewards_all, avgs_rewards_partial,
                        avgs_frames_all, avgs_frames_partial,
                        dict_sizes,
                        mins_feat_weights, maxs_feat_weights, avgs_feat_weights,
                        filename_prefix)

        ale.reset_game()

    logging.info('Ending training')
    file_utils.save_weights(agent.weights, filename)

if __name__ == '__main__':
    game = GAME + '.bin'
    gamepath = os.path.join('roms', game)

    # load the ale interface to interact with
    ale = ALEInterface()
    ale.setInt('random_seed', 42)
    ale.setFloat("repeat_action_probability", 0.00)
    ale.setInt("frame_skip", config['frame_skip'])
    ale.setBool("color_averaging", True)
    ale.loadROM(gamepath)

    feature_extractor = feature_extractors.AtariFeatureExtractor(mode=FEATURES,
                                background=file_utils.load_background(GAME))

    agent = learning_agents.SARSALambdaLearningAlgorithm(
                actions=ale.getMinimalActionSet(),
                featureExtractor=feature_extractor,
                discount=config['gamma'],
                explorationProb=config['exploration_probability'],
                stepSize=config['step'],
                decay=config['lambda'] * config['gamma'],
                threshold=config['elegibility_traces_threshold'])

    logging.basicConfig(filename='logs/{}-{}-{}.log'.format(GAME, agent.name, FEATURES),
                        format='%(asctime)s %(message)s', level=logging.INFO)

    if LOAD_WEIGHTS:
        agent.weights = file_utils.load_weights(WEIGHTS_FILENAME)

    train_agent(ale, agent)

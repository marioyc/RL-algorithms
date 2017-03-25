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
FEATURES = 'basic'
f = open('config.json')
config = json.load(f)[FEATURES]

# training parameters
GAME = 'space_invaders'
LOAD_WEIGHTS = False
LOAD_WEIGHTS_FILENAME = ''
NUM_EPISODES_AVERAGE_OVER = 30
TEST_INTERVAL = 50
NUM_EPISODES_TEST_OVER = 30
RECORD_BEST = True

random.seed(42)

def run_episode(ale, agent, train=True):
    total_reward = 0
    num_frames = 0
    newAction = random.choice(agent.actions)

    frames = []
    screen = ale.getScreen()
    state = {"screen" : screen}
    agent.startEpisode(state)
    initial_value = agent.getValue()

    while not ale.game_over():
        # if newAction is None then we are training an off-policy algorithm
        # otherwise, we are training an on policy algorithm
        if not train or newAction is None:
            action = agent.getAction()
        else:
            action = newAction

        reward = ale.act(action)
        total_reward += reward

        if not ale.game_over():
            new_screen = ale.getScreen()
            if RECORD_BEST:
                frames.append(ale.getScreenRGB())
            new_state = {"screen": new_screen}
        else:
            new_state = None

        if train:
            newAction = agent.incorporateFeedback(state, action, reward, new_state)
        elif new_state is not None:
            agent.featureExtractor.extractFeatures(new_state)

        state = new_state
        num_frames += 1

    ale.reset_game()
    return initial_value, total_reward, num_frames, frames

def train_agent(ale, agent):
    """
    trains an agent to play a game
    ale: instance of the ALE interface
    agent: the algorithm/agent that learns to play the game
    """

    screen_dims = ale.getScreenDims()
    assert(screen_dims[0] == 160 and screen_dims[1] == 210)

    # statistics
    stats = {
        "average_interval" : NUM_EPISODES_AVERAGE_OVER,
        "rewards" : [],
        "rewards_average_all" : [],
        "rewards_average_partial" : [],
        "initial_value" : [],
        "frames" : [],
        "frames_average_all" : [],
        "frames_average_partial" : [],
        "features" : [],
        "feature_weights_min" : [],
        "feature_weights_max" : [],
        "feature_weights_average" : [],
        "test_interval": TEST_INTERVAL,
        "test_mean" : [],
        "test_std" : [],
    }
    best_reward = 0

    # flag for first non-zero reward
    sawFirst = False
    firstReward = 0.0

    logging.info('Starting training')
    for episode in tqdm(range(config['train_episodes'])):
        start = time.time()
        initial_value, total_reward, num_frames, frames = run_episode(ale, agent)
        end = time.time()

        logging.info('episode: %d, score: %d, number of frames: %d, time: %.4fm',
                    episode, total_reward, num_frames, (end - start) / 60)

        filename_prefix = "{}-{}-{}".format(GAME, agent.name, FEATURES)
        filename = "{}-{}".format(filename_prefix,  episode)

        if total_reward > best_reward:
            best_reward = total_reward
            logging.info('Best reward: %d', total_reward)

            if RECORD_BEST:
                file_utils.save_videos(frames, screen_dims, filename)
                file_utils.save_weights(agent.weights, filename)

        # update and plot statistics of current episode
        stats["rewards"].append(total_reward)
        stats["rewards_average_all"].append(np.mean(stats["rewards"]))
        stats["rewards_average_partial"].append(np.mean(stats["rewards"][-NUM_EPISODES_AVERAGE_OVER:]))

        stats["initial_value"].append(initial_value)

        stats["frames"].append(num_frames)
        stats["frames_average_all"].append(np.mean(stats["frames"]))
        stats["frames_average_partial"].append(np.mean(stats["frames"][-NUM_EPISODES_AVERAGE_OVER:]))

        stats["features"].append(len(agent.weights))

        weights = [v for k,v in agent.weights.iteritems()]
        stats["feature_weights_min"].append(min(weights))
        stats["feature_weights_max"].append(max(weights))
        stats["feature_weights_average"].append(np.mean(weights))

        if (episode + 1) % TEST_INTERVAL == 0:
            test_results = np.zeros(NUM_EPISODES_TEST_OVER)
            for test_episode in range(NUM_EPISODES_TEST_OVER):
                initial_value, total_reward, num_frames, frames = run_episode(ale, agent, train=False)
                logging.info('test episode: %d, score: %d, number of frames: %d',
                            test_episode, total_reward, num_frames)
                test_results[test_episode] = total_reward
            stats["test_mean"].append(np.mean(test_results))
            stats["test_std"].append(np.std(test_results))

        file_utils.plot_stats(stats, filename_prefix)

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
    #ale.setBool("color_averaging", True)
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

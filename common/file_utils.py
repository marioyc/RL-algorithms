"""
:description: save and load weight vectors and theano models
"""
import os
import csv
import pickle
import cPickle
import datetime

import numpy as np

WEIGHTS_DIR = "weights"
STATS_DIR = "stats"
MODELS_DIR = "models"
BACKGROUNDS_DIR = "backgrounds"

def save_weights(weights, filename):
    output_filepath = os.path.join(WEIGHTS_DIR, "{}.pkl".format(filename))
    with open(output_filepath, 'wb') as f:
        pickle.dump(weights, f)

def load_weights(filename):
    print "loading weights..."
    filepath = os.path.join(WEIGHTS_DIR, filename)
    weights = {}
    try:
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
    except IOError as e:
        print("!!!weight file {} not found, reinitializing weights!!!".format(filepath))
        raise(e)
    return weights

def save_model(model, model_filename):
    f = file(os.path.join(MODELS_DIR, model_filename), 'wb')
    try:
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    except:
        print('could not save model')
    f.close()

def load_model(model_filename):
    f = file(os.path.join(MODELS_DIR, model_filename), 'rb')
    model = cPickle.load(f)
    f.close()
    return model

def save_stats(rewards, avg_rewards_all, avg_rewards_partial, dict_sizes,
                min_weights, max_weights, avg_weights, num_frames, filename):
    filepath = os.path.join(STATS_DIR, "{}".format(filename))
    np.savez(filepath, rewards=rewards, avg_rewards_all=avg_rewards_all,
            avg_rewards_partial=avg_rewards_partial, dict_sizes=dict_sizes,
            min_weights=min_weights, max_weights=max_weights,
            avg_weights=avg_weights, num_frames=num_frames)

def load_stats(filename):
    filepath = os.path.join(STATS_DIR, "{}".format(filename))
    return np.load(filepath)

def load_background(game):
    f = file(os.path.join(BACKGROUNDS_DIR, "{}.bg".format(game)), 'rb')
    w, h = [int(x) for x in f.readline()[:-1].split(',')]
    background = []
    print w, h
    for i in range(0,h):
        line = f.readline()[:-1]
        background.append([int(x) for x in line.split(',')])
    print background

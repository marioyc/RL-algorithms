import cPickle
import csv
#import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

WEIGHTS_DIR = "weights"
STATS_DIR = "stats"
BACKGROUNDS_DIR = "backgrounds"
VIDEOS_DIR = "videos"

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
        print("weight file {} not found".format(filepath))
        raise(e)
    return weights

def save_stats(rewards, avg_rewards_all, avg_rewards_partial, dict_sizes,
                min_weights, max_weights, avg_weights, num_frames,
                avg_frames_all, avg_frames_partial, filename):
    filepath = os.path.join(STATS_DIR, "{}".format(filename))
    np.savez(filepath, rewards=rewards, avg_rewards_all=avg_rewards_all,
            avg_rewards_partial=avg_rewards_partial, dict_sizes=dict_sizes,
            min_weights=min_weights, max_weights=max_weights,
            avg_weights=avg_weights, num_frames=num_frames,
            avg_frames_all=avg_frames_all, avg_frames_partial=avg_frames_partial)

def load_stats(filename):
    filepath = os.path.join(STATS_DIR, "{}".format(filename))
    return np.load(filepath)

def plot_stats(avg_rewards_all, avg_rewards_partial,
            avg_frames_all, avg_frames_partial,
            dict_sizes, min_weights, max_weights, avg_weights,
            filename):
    x_axis = range(1, len(avg_rewards_all) + 1)
    filepath = os.path.join(STATS_DIR, filename)

    f, axarr = plt.subplots(2, sharex=True)

    axarr[0].set_ylabel('Total reward average')
    avg_all, = axarr[0].plot(x_axis, avg_rewards_all, label='all')
    avg_partial, = axarr[0].plot(x_axis, avg_rewards_partial, label='last 50')
    legend1 = axarr[0].legend(bbox_to_anchor=(1, 1), loc=2, ncol=1,prop={'size':10})

    axarr[1].set_ylabel('Number of frames average')
    axarr[1].plot(x_axis, avg_frames_all, label='all')
    axarr[1].plot(x_axis, avg_frames_partial, label='last 50')
    legend2 = axarr[1].legend(bbox_to_anchor=(1, 1), loc=2, ncol=1,prop={'size':10})

    f.savefig(filepath + '-rewards_frames.png', bbox_extra_artists=(legend1, legend2), bbox_inches='tight')

    f, axarr = plt.subplots(2, sharex=True)

    axarr[0].set_ylabel('Number of weights')
    axarr[0].plot(x_axis, dict_sizes)

    axarr[1].set_ylabel('Range of weights')
    axarr[1].plot(x_axis, min_weights)
    axarr[1].plot(x_axis, max_weights)
    axarr[1].plot(x_axis, avg_weights)

    f.savefig(filepath + '-weights.png', bbox_extra_artists=(legend1, legend2), bbox_inches='tight')

def load_background(game):
    f = file(os.path.join(BACKGROUNDS_DIR, "{}.bg".format(game)), 'rb')
    w, h = [int(x) for x in f.readline()[:-1].split(',')]
    background = []
    for i in range(0,h):
        line = f.readline()[:-1]
        background.append([int(x) / 2 for x in line.split(',')])

def save_videos(video_frames, screen_dims, filename):
    filepath = os.path.join(VIDEOS_DIR, "{}.avi".format(filename))
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter(filepath, fourcc, 24, screen_dims)
    for frame in video_frames:
        video.write(frame)
    video.release()

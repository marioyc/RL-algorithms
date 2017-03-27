import cPickle
import cv2
import numpy as np
import os
import pickle

#http://stackoverflow.com/questions/27147300/how-to-clean-images-in-python-django
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def plot_stats(stats, filename):
    x_axis = range(1, len(stats["rewards"]) + 1)
    filepath = os.path.join(STATS_DIR, filename)

    f, axarr = plt.subplots(3, sharex=True, figsize=(8, 15))

    axarr[0].set_ylabel('Total reward average')
    axarr[0].plot(x_axis, stats["rewards_average_all"], label='all')
    axarr[0].plot(x_axis, stats["rewards_average_partial"], label="last {}".format(stats["average_interval"]))
    legend1 = axarr[0].legend(bbox_to_anchor=(1, 1), loc=2, ncol=1,prop={'size':10})

    axarr[1].set_ylabel('Value of first frame')
    axarr[1].plot(x_axis, stats["initial_value"])

    axarr[2].set_ylabel('Number of frames average')
    axarr[2].plot(x_axis, stats["frames_average_all"], label='all')
    axarr[2].plot(x_axis, stats["frames_average_partial"], label="last {}".format(stats["average_interval"]))
    legend2 = axarr[2].legend(bbox_to_anchor=(1, 1), loc=2, ncol=1,prop={'size':10})

    f.savefig(filepath + '-rewards_value_frames.png', bbox_extra_artists=(legend1, legend2), bbox_inches='tight')
    plt.close(f)

    f, axarr = plt.subplots(2, sharex=True, figsize=(8, 8))

    axarr[0].set_ylabel('Number of weights')
    axarr[0].plot(x_axis, stats["features"])

    axarr[1].set_ylabel('Range of weights')
    axarr[1].plot(x_axis, stats["feature_weights_min"])
    axarr[1].plot(x_axis, stats["feature_weights_max"])
    axarr[1].plot(x_axis, stats["feature_weights_average"])

    f.savefig(filepath + '-weights.png', bbox_inches='tight')
    plt.close(f)

    if stats["test_mean"]:
        x_axis = np.arange(1, 1 + len(stats["test_mean"])) * stats["test_interval"]
        f, ax = plt.subplots(1, sharex=True, figsize=(8, 6))
        ax.errorbar(x_axis, stats["test_mean"], stats["test_std"], marker='^')
        f.savefig(filepath + '-test.png', bbox_inches='tight')
        plt.close(f)

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

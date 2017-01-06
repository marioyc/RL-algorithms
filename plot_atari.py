import matplotlib.pyplot as plt

import common.file_utils as file_utils

GAMES = ['alien', 'star_gunner', 'video_pinball', 'atlantis', 'centipede']
AGENT = 'SARSALambda'
FEATURES = 'bpros'

for game in GAMES:
    stats = file_utils.load_stats('{}-{}-{}.npz'.format(game, AGENT, FEATURES))

    f, axarr = plt.subplots(2, sharex=True)

    #rewards, = axarr[0].plot(stats['rewards'], label='reward')
    avg_all, = axarr[0].plot(stats['avg_rewards_all'], label='all')
    avg_partial, = axarr[0].plot(stats['avg_rewards_partial'], label='last 50')
    legend1 = axarr[0].legend(bbox_to_anchor=(1, 1), loc=2, ncol=1,prop={'size':10})

    #axarr[1].plot(stats['num_frames'], label='frames')
    axarr[1].plot(stats['avg_frames_all'], label='all')
    axarr[1].plot(stats['avg_frames_partial'], label='last 50')
    legend2 = axarr[1].legend(bbox_to_anchor=(1, 1), loc=2, ncol=1,prop={'size':10})

    f.savefig('stats/' + game + '-' + AGENT + '-' + FEATURES + '-rewards_frames.png', bbox_extra_artists=(legend1, legend2), bbox_inches='tight')

    f, axarr = plt.subplots(2, sharex=True)

    axarr[0].plot(stats['dict_sizes'])

    axarr[1].plot(stats['min_weights'])
    axarr[1].plot(stats['max_weights'])
    axarr[1].plot(stats['avg_weights'])

    f.savefig('stats/' + game + '-' + AGENT + '-' + FEATURES + '-weights.png', bbox_extra_artists=(legend1, legend2), bbox_inches='tight')

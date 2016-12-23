import matplotlib.pyplot as plt

import common.file_utils as file_utils

GAME = 'space_invaders'
AGENT = 'SARSALambda'

stats = file_utils.load_stats('{}-{}.npz'.format(GAME, AGENT))

f, axarr = plt.subplots(2, sharex=True)

#rewards, = axarr[0].plot(stats['rewards'], label='reward')
avg_all, = axarr[0].plot(stats['avg_rewards_all'], label='all')
avg_partial, = axarr[0].plot(stats['avg_rewards_partial'], label='last 50')
legend1 = axarr[0].legend(bbox_to_anchor=(1, 1), loc=2, ncol=1,prop={'size':10})

#axarr[1].plot(stats['num_frames'], label='frames')
axarr[1].plot(stats['avg_frames_all'], label='all')
axarr[1].plot(stats['avg_frames_partial'], label='last 50')
legend2 = axarr[1].legend(bbox_to_anchor=(1, 1), loc=2, ncol=1,prop={'size':10})

f.savefig('stats/' + GAME + '_' + AGENT + '_rewards_frames.png', bbox_extra_artists=(legend1, legend2), bbox_inches='tight')

f, axarr = plt.subplots(2, sharex=True)

axarr[0].plot(stats['dict_sizes'])

axarr[1].plot(stats['min_weights'])
axarr[1].plot(stats['max_weights'])
axarr[1].plot(stats['avg_weights'])

f.savefig('stats/' + GAME + '_' + AGENT + '_weights.png', bbox_extra_artists=(legend1, legend2), bbox_inches='tight')

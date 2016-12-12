import matplotlib.pyplot as plt

import common.file_utils as file_utils

stats = file_utils.load_stats('SARSALambda-stats.npz')

f, axarr = plt.subplots(4, sharex=True)

rewards, = axarr[0].plot(stats['rewards'], label='reward')
avg_all, = axarr[0].plot(stats['avg_rewards_all'], label='average(all)')
avg_partial, = axarr[0].plot(stats['avg_rewards_partial'], label='average(last 50)')
axarr[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)

axarr[1].plot(stats['dict_sizes'])

axarr[2].plot(stats['min_weights'])
axarr[2].plot(stats['max_weights'])
axarr[2].plot(stats['avg_weights'])

axarr[3].plot(stats['num_frames'])

plt.show()

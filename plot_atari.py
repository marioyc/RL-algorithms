import matplotlib.pyplot as plt

import common.file_utils as file_utils

stats = file_utils.load_stats('SARSALambda-stats.npz')

f, axarr = plt.subplots(5, sharex=True)
axarr[0].plot(stats['rewards'])
axarr[1].plot(stats['avg_rewards_all'])
axarr[1].plot(stats['avg_rewards_partial'])
axarr[2].plot(stats['dict_sizes'])
axarr[3].plot(stats['min_weights'])
axarr[3].plot(stats['max_weights'])
axarr[3].plot(stats['avg_weights'])
axarr[4].plot(stats['num_frames'])
plt.show()

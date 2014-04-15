from __future__ import division

import zstackLib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11})

filename = '/home/brian/local/drumCreep/20x_gfp_stack_t2.tif'
z1 = zstackLib.zstack(filename)
z1.focusScan(threshValue=127,blurWindow=5,particleAnalysis=0)


fig=plt.figure(figsize=(4,3))
ax1 = fig.add_subplot(111)
fig.subplots_adjust(top=0.95, bottom=0.17, left=0.17, right=0.95)
ax1.plot(z1.sharpness)
ax1.set_xlabel('Frame')
ax1.set_ylabel('Mean high frequency content')

plt.savefig('zstackAnalysis.eps', dpi=160, facecolor='w')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

import pickle
with open('test_2_eeg','rb') as f:
	data = pickle.load(f)
print(data.shape)
num_samples = data[0,:].size

print(num_samples)

#fig,(ax1,ax2)= plt.subplots(2,1,sharex='all',sharey='none')
plt.plot(data[0,:])
#ax2.plot(data[1,10:].T)

#ax2.set_xlabel('s')
#ax1.set_ylabel('V')
#ax2.set_ylabel('V')
#xticks = np.arange(0, data[0,:].size,1000)
#ax2.set_xticks(xticks)
#ax2.set_xticklabels(xticklabels)
plt.show()

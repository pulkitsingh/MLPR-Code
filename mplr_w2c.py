# Plotting sums of samples from an exponential distribution to demonstrate
# Central Limit Theorem

import numpy as np 
import matplotlib.pyplot as plt 

N = 100000
K = 10
expSums = []

# Sampling K numbers from exponential distribution, adding them up
# Plotting N sums to observe shape
for i in range(0, N):
	expSums.append(float(sum(np.random.exponential(0.5, [K,1]))[0]))

hist_sums = plt.hist(expSums, bins=100)
plt.show()
# Demonstrating gaussian distribution by sampling & plotting uniform random
# numbers

import numpy as np
import matplotlib.pyplot as plt

N = int(1e6) # 1e6 is a float, numpy wants int arguments
xx = np.random.randn(N)
hist_stuff = plt.hist(xx, bins=100)

print('empirical_mean = %g' % np.mean(xx)) # or xx.mean()
print('empirical_var = %g' % np.var(xx))   # or xx.var()


bin_centres = 0.5*(hist_stuff[1][1:] + hist_stuff[1][:-1])
# Fill in an expression to evaluate the PDF at the bin_centres.
# To square every element of an array, use **2
mu = np.mean(xx)
sigma = np.var(xx)
pdf = np.exp(-0.5*(bin_centres**2)) / np.sqrt(2*np.pi);

bin_width = bin_centres[1] - bin_centres[0]
predicted_bin_heights = pdf * N * bin_width
# Finally, plot the theoretical prediction over the histogram:
plt.plot(bin_centres, predicted_bin_heights, '-r')
plt.show()

import numpy as np
import matplotlib.pyplot as plt 

class1 = [0.5,0.1,0.2,0.4,0.3,0.2,0.2,0.1,0.35,0.25]
class2 = [0.9,0.8,0.75,1.0]

# calculating empirical mean and variances to fit 1D gaussians
c1_mean = np.mean(class1); c1_var = np.var(class1)
c2_mean = np.mean(class2); c2_var = np.var(class2)

print c1_mean, c1_var
print c2_mean, c2_var

# function to evaluate gaussian pdf at particular input
def norm(x, mean, var):
	scale = 1 / np.sqrt(2 * np.pi * var)
	return scale * np.exp(-np.square(x - mean) / (2 * var))


x_grid = np.arange(-0.5,1.5, 0.01)[:, None]

# Calculating class probabilities
c1_prob = float(len(class1)) / float((len(class1) + len(class2)))
c2_prob = 1 - c1_prob


# plotting p(x,y) = P(y) * p(x|y) for each class
plt.clf()
plt.plot(x_grid, norm(x_grid, c1_mean, c1_var) * c1_prob, '-b')
plt.plot(x_grid, norm(x_grid, c2_mean, c2_var) * c2_prob, '-r')
plt.plot([0.6]*300, np.arange(0, 3, 0.01), '-g')
plt.show()

# calculating P(y = 1 | x = 0.6)

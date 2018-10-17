# Regressing on dummy binary classification data to understand why 
# it can be a bad idea

import numpy as np 
import matplotlib.pyplot as plt 

# train model on synthetic dataset
N = 200
X = np.random.randn(N, 1)*10
yy = (X>1) & (X<3)
def phi_fn(X):
	return np.concatenate([np.ones((X.shape[0],1)), X, X**2], axis=1)
ww = np.linalg.lstsq(phi_fn(X), yy)[0]

# predictions
x_grid = np.arange(0, 10, 0.05)[:, None]
f_grid = np.dot(phi_fn(x_grid), ww)

# predictions with alternate weights
w2 = [-1, 2, -0.5] # values set by hand
f2_grid = np.dot(phi_fn(x_grid), w2)

# show demo
plt.clf()
plt.plot(X[yy==1], yy[yy==1], 'r+')
plt.plot(X[yy==0], yy[yy==0], 'bo')
plt.plot(x_grid, f_grid, 'y-')
plt.plot(x_grid, f2_grid, 'g-')
plt.ylim([-0.1, 1.1])
plt.show()
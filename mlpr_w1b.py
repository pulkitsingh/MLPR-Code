import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------#

# cooking up some input, weights and output
X = np.array([[1, 4, 3],[4, 2, 7],[7, 4, 9]])
yy = np.array([5, 20, 35])

# using least squares to find optimal fit
#w_fit = np.linalg.lstsq(X, yy)[0]
#print w_fit

# adding a bias term
X_bias = np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1)
w_bias_fit = np.linalg.lstsq(X, yy)[0]
print w_bias_fit


#----------------------------------------------------------------------------#

# plotting the cos function 

grid_size = 0.1
x_grid = np.arange(-10, 10, grid_size)
f_vals = np.cos(x_grid)
plt.clf()
plt.plot(x_grid, f_vals, 'b-')
plt.plot(x_grid, f_vals, 'r.')
plt.show()

#----------------------------------------------------------------------------#

# Plotting 1D Radial Basis Function to understand effect of parameters

def rbf_1d(xx, cc, hh):
	return np.exp(- (xx - cc)**2 / hh**2)

plt.clf()
grid_size = 0.01
x_grid = np.arange(-10, 10, grid_size)
plt.plot(x_grid, rbf_1d(x_grid, cc=5, hh=1), '-b')
plt.plot(x_grid, rbf_1d(x_grid, cc=-2, hh=2), '-r')
plt.show() # I may forget sometimes. Not necessary in python --pylab

#----------------------------------------------------------------------------#

# Plotting 1D Logistic Sigmoid Function to understand effect of parameters

def sigmoid_1d(xx, vv, bb):
	return 1 / (1 + np.exp(-((vv * xx) + bb)))

plt.clf()
grid_size = 0.01
x_grid = np.arange(-10, 10, grid_size)
plt.plot(x_grid, sigmoid_1d(x_grid, vv=0.5, bb=0), '-b')
plt.plot(x_grid, sigmoid_1d(x_grid, vv=4, bb=2), '-r')
plt.show()

#----------------------------------------------------------------------------#


plt.clf()
grid_size = 0.01
x_grid = np.arange(-10, 10, grid_size)
plt.plot(x_grid, 
	     2 * rbf_1d(x_grid, cc=-5, hh=1) - rbf_1d(x_grid, cc=5, hh=1), '-b')
plt.show()

#----------------------------------------------------------------------------#

# Plotting different fits on dummy data

# Set up and plot the dataset
yy = np.array([1.1, 2.3, 2.9]) # N,
X = np.array([[0.8], [1.9], [3.1]]) # N,1
plt.clf()
plt.plot(X, yy, 'x', MarkerSize=20, LineWidth=2)

# phi-functions to create various matrices of new features from an original 
# matrix of 1D inputs.
def phi_linear(Xin):
    return np.hstack([np.ones((Xin.shape[0],1)), Xin])
def phi_quadratic(Xin):
    return np.hstack([np.ones((Xin.shape[0],1)), Xin, Xin**2])
def fw_rbf(xx, cc):
    return np.exp(-(xx-cc)**2 / 2.0)
def phi_rbf(Xin):
    return np.hstack([fw_rbf(Xin, 1), fw_rbf(Xin, 2), fw_rbf(Xin, 3)])

def fit_and_plot(phi_fn, X, yy):
    # phi_fn takes N, inputs and returns N,D basis function values
    w_fit = np.linalg.lstsq(phi_fn(X), yy)[0] # D,
    X_grid = np.arange(0, 4, 0.01)[:,None] # N,1
    f_grid = np.dot(phi_fn(X_grid), w_fit)
    plt.plot(X_grid, f_grid, LineWidth=2)

fit_and_plot(phi_linear, X, yy)
fit_and_plot(phi_quadratic, X, yy)
fit_and_plot(phi_rbf, X, yy)
plt.legend(('data', 'linear fit', 'quadratic fit', 'rbf fit'))

plt.show()

#----------------------------------------------------------------------------#




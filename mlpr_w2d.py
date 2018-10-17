# Computing standard error on the mean on a toy bernoulli distribution

import numpy as np
import matplotlib.pyplot as plt

N = 100
bernoulli = np.random.rand(1, N) < 0.3
mean = np.mean(bernoulli)
sd = np.std(bernoulli)
error = sd / np.sqrt(N)

print('Mean = %g' % mean)
print('Standard Deviation = %g' % sd)
print('Standard Error = %g' % error) 


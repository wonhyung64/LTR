#%%
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp


#%%
mu_1 = (1,2,3,4)
cov = [[]]
np.reshape([])

dim = 4
diagonal = np.random.uniform(0.1, 1.0, dim)
off_diagonal = np.random.uniform(-0.5, 0.5, size=(dim, dim))
cov_1 = np.diag(diagonal) + off_diagonal + off_diagonal.T

# rv = sp.stats.multivariate_normal(mu_1, cov_1)


step = 0
cond_samples = []
while True:
    step += 1
    sample = np.random.multivariate_normal(mu_1, cov_1, 1)

    [[z1, z2, z3, z4]] = sample.tolist()
    if z1 > z2 > z3 > z4:
        cond_samples.append(sample)
        if len(cond_samples) == 50:
            break
    print(step)



samples = np.concatenate(cond_samples, axis=0)
np.mean(samples, axis=0)
np.var(samples, axis=0)
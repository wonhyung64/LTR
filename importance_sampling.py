#%%
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp


def pdf_1(z: np.ndarray, mu_1: np.ndarray, cov_1: np.ndarray):
    return 1 / (((2 * np.math.pi)**2) * np.sqrt(np.linalg.det(cov_1))) \
    *np.exp(-1/2 * (z - mu_1).T @ np.linalg.inv(cov_1) @ (z - mu_1))

def pdf_2():
    pass

def indicator_c(z):
    [z1,z2,z3,z4] = z.tolist()
    return z1 > z2 > z3 > z4


#%%
mu_1 = np.array((1,2,3,4))
z
cov_1 = np.array([
    [9,3,4,5],
    [3,7,2,1],
    [4,2,5,2],
    [5,1,2,6]
    ])


cov_2 = np.array([
    [4,.25,.25,.25],
    [.25,4,.25,.25],
    [.25,.25,4,.25],
    [.25,.25,.25,4]
    ])

seeds = range(10)
max_sampling_list = [50, 100, 200]
seed = 0
max_sampling_num = 50
for seed in seeds:
    for max_sampling_num in max_sampling_list:
        sampling_num = 0
        cond_samples = []
        while True:
            sampling_num += 1
            z = np.random.multivariate_normal(mu_1, cov_2, 1)

            [[z1, z2, z3, z4]] = z.tolist()
            if z1 > z2 > z3 > z4:
                cond_samples.append(z)
                if len(cond_samples) == max_sampling_num:
                    break
            # print(step)
sampling_num
len(cond_samples)
a = np.array([[1,2,3,], [2,3,4,]])
np.exp(a)
def g1_fn(z):
    return 1



samples = np.concatenate(cond_samples, axis=0)
sum([g1_fn(samples[i]) * pdf_1(samples[i], mu_1, cov_2) for i in range(len(samples))]) / len(samples)

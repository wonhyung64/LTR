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
mu_2 = np.array((4,3,2,1))
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


#%%
seeds = 30
max_sampling_num = 10000

mc_seed_100 = []
is_seed_100 = []
mc_seed_1000 = []
is_seed_1000 = []
mc_seed_10000 = []
is_seed_10000 = []

for seed in range(seeds):
    print(f"seed: {seed}")
    np.random.seed(seed=seed)
    mc_samples = []
    is_samples = []
    for sampling_num in range(max_sampling_num):
        z = np.random.multivariate_normal(mu_1, cov_2, 1)
        mc_samples.append(z)
        is_z = np.random.multivariate_normal(mu_2, cov_2, 1)
        is_samples.append(is_z)
        if sampling_num+1 == 100:
            mc_mu = sum([indicator_c(z) for z in np.concatenate(mc_samples)]) / 100
            mc_seed_100.append(mc_mu)
            is_mu = sum([indicator_c(z)*pdf_1(z,mu_1,cov_2)/pdf_1(z,mu_2,cov_2) for z in np.concatenate(is_samples)]) / 100
            is_seed_100.append(is_mu)
        if sampling_num+1 == 1000:
            mc_mu = sum([indicator_c(z) for z in np.concatenate(mc_samples)]) / 1000
            mc_seed_1000.append(mc_mu)
            is_mu = sum([indicator_c(z)*pdf_1(z,mu_1,cov_2)/pdf_1(z,mu_2,cov_2) for z in np.concatenate(is_samples)]) / 1000
            is_seed_1000.append(is_mu)
        if sampling_num+1 == 10000:
            mc_mu = sum([indicator_c(z) for z in np.concatenate(mc_samples)]) / 10000
            mc_seed_10000.append(mc_mu)
            is_mu = sum([indicator_c(z)*pdf_1(z,mu_1,cov_2)/pdf_1(z,mu_2,cov_2) for z in np.concatenate(is_samples)]) / 10000
            is_seed_10000.append(is_mu)


#%%
np.mean(mc_seed_100[:10])
np.mean(is_seed_100[:10])
np.var(mc_seed_100[:10], ddof=1)
np.var(is_seed_100[:10], ddof=1)

np.mean(mc_seed_1000[:10])
np.mean(is_seed_1000[:10])
np.var(mc_seed_1000[:10], ddof=1)
np.var(is_seed_1000[:10], ddof=1)

np.mean(mc_seed_10000[:10])
np.mean(is_seed_10000[:10])
np.var(mc_seed_10000[:10], ddof=1)
np.var(is_seed_10000[:10], ddof=1)

np.mean(mc_seed_100)
np.mean(is_seed_100)
np.var(mc_seed_100, ddof=1)
np.var(is_seed_100, ddof=1)

np.mean(mc_seed_1000)
np.mean(is_seed_1000)
np.var(mc_seed_1000, ddof=1)
np.var(is_seed_1000, ddof=1)

np.mean(mc_seed_10000)
np.mean(is_seed_10000)
np.var(mc_seed_10000, ddof=1)
np.var(is_seed_10000, ddof=1)


#%%
seed = 30

mc_seed_50 = []
is_seed_50 = []
mc_seed_100 = []
is_seed_100 = []
mc_seed_200 = []
is_seed_200 = []

for seed in range(seed):
    print(f"seed: {seed}")
    np.random.seed(seed=seed)
    cond_mc_num = 0
    cond_is_num = 0
    sampling_num = 0
    while True:
        sampling_num += 1

        if cond_mc_num < 200:
            z = np.random.multivariate_normal(mu_1, cov_2, 1)
            [[z1, z2, z3, z4]] = z.tolist()

            if z1 > z2 > z3 > z4:
                cond_mc_num += 1
                if cond_mc_num == 50:
                    mc_seed_50.append(sampling_num)
                if cond_mc_num == 100:
                    mc_seed_100.append(sampling_num)
                if cond_mc_num == 200:
                    mc_seed_200.append(sampling_num)

        if cond_is_num < 200:
            z = np.random.multivariate_normal(mu_2, cov_2, 1)
            [[z1, z2, z3, z4]] = z.tolist()

            if z1 > z2 > z3 > z4:
                cond_is_num += 1
                if cond_is_num == 50:
                    is_seed_50.append(sampling_num)
                if cond_is_num == 100:
                    is_seed_100.append(sampling_num)
                if cond_is_num == 200:
                    is_seed_200.append(sampling_num)

        if cond_mc_num == 200 and cond_is_num == 200:
            break


#%%
np.mean(mc_seed_50[:10])
np.mean(is_seed_50[:10])
np.mean(mc_seed_100[:10])
np.mean(is_seed_100[:10])
np.mean(mc_seed_200[:10])
np.mean(is_seed_200[:10])

np.mean(mc_seed_50)
np.mean(is_seed_50)
np.mean(mc_seed_100)
np.mean(is_seed_100)
np.mean(mc_seed_200)
np.mean(is_seed_200)
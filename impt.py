#%%
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def result2df(ns_seed_result, is_seed_result):
    mu_rows = ns_seed_result + is_seed_result
    num_of_samples = len(ns_seed_result)
    sampling_rows = ["Naive"]*num_of_samples + ["Importance"]*num_of_samples
    result_df = pd.DataFrame(
        data=np.array([mu_rows, sampling_rows]).T,
        columns=["mu", "sampling"]
    )
    result_df["mu"] = result_df["mu"].astype(float)
    result_df["sampling"] = result_df["sampling"].astype(object)

    return result_df


def pdf(z: np.ndarray, mu: np.ndarray, cov: np.ndarray):
    frac_term = 1 / (((2 * np.math.pi)**2) * np.sqrt(np.linalg.det(cov)))
    exp_term = np.exp(-1/2 * (z - mu).T @ np.linalg.inv(cov) @ (z - mu))

    return  frac_term * exp_term


def indicator_c(z):
    [z1,z2,z3,z4] = z.tolist()

    return z1 > z2 > z3 > z4


def show_result(mc_result, is_result, num_seed):
    print(f"NS mean: {np.mean(mc_result[:num_seed])}")
    print(f"IS mean: {np.mean(is_result[:num_seed])}")
    print(f"NS var: {np.var(mc_result[:num_seed], ddof=1)}")
    print(f"IS var: {np.var(is_result[:num_seed], ddof=1)}")


def likelihood_ratio(z, mu_1, mu_2, cov_1, cov_2):
    return pdf(z, mu_1, cov_1) / pdf(z, mu_2, cov_2)


#%%
"""setup"""
mu_1 = np.array((1,2,3,4))
mu_2 = np.array((4,3,2,1))

diagonal = 4.
off_diagonal = .25

cov = np.ones([4, 4]) * off_diagonal
np.fill_diagonal(cov, diagonal, wrap=True)

cov2 = np.ones([4, 4]) * off_diagonal
# scale_factor = 1.
# np.fill_diagonal(cov2, diagonal * scale_factor, wrap=True)

seeds = 1000
max_sampling_num = 100000
scale_factors = np.arange(0.5, 2.3, 0.3)

save_dir = "./data/simulation"
scale_factor = 1.

print("IS")
print(datetime.datetime.now())
"""importance sampling의 분산 조절"""
np.fill_diagonal(cov2, diagonal * scale_factor, wrap=True)
is_seed_100 = []
is_seed_1000 = []
is_seed_10000 = []
is_seed_100000 = []

for seed in range(seeds):
    """seed 반복실험"""
    print(f"seed: {seed}")
    np.random.seed(seed=seed)

    is_samples = []
    for sampling_num in range(max_sampling_num):
        """샘플링"""
        is_z = np.random.multivariate_normal(mu_2, cov2, 1)
        is_samples.append(is_z)

        if sampling_num+1 == 100:
            """샘플링 개수 100개인 경우의 추정치"""
            is_mu = np.mean([indicator_c(z) * likelihood_ratio(z, mu_1, mu_2, cov, cov2) for z in np.concatenate(is_samples)])
            is_seed_100.append(is_mu)

        if sampling_num+1 == 1000:
            """샘플링 개수 1000개인 경우의 추정치"""
            is_mu = np.mean([indicator_c(z) * likelihood_ratio(z, mu_1, mu_2, cov, cov2) for z in np.concatenate(is_samples)])
            is_seed_1000.append(is_mu)

        if sampling_num+1 == 10000:
            """샘플링 개수 10000개인 경우의 추정치"""
            is_mu = np.mean([indicator_c(z) * likelihood_ratio(z, mu_1, mu_2, cov, cov2) for z in np.concatenate(is_samples)])
            is_seed_10000.append(is_mu)

        if sampling_num+1 == 100000:
            """샘플링 개수 100000개인 경우의 추정치"""
            is_mu = np.mean([indicator_c(z) * likelihood_ratio(z, mu_1, mu_2, cov, cov2) for z in np.concatenate(is_samples)])
            is_seed_100000.append(is_mu)

    # is_seed_samples.append(np.concatenate(is_samples))

# is_seed_samples_arr = np.stack(is_seed_samples, axis=0)
scale_factor_str = "_".join(str(scale_factor).split("."))

np.save(f"{save_dir}/is_100_{scale_factor_str}", np.array(is_seed_100), allow_pickle=True)
np.save(f"{save_dir}/is_1000_{scale_factor_str}", np.array(is_seed_1000), allow_pickle=True)
np.save(f"{save_dir}/is_10000_{scale_factor_str}", np.array(is_seed_10000), allow_pickle=True)
np.save(f"{save_dir}/is_100000_{scale_factor_str}", np.array(is_seed_100000), allow_pickle=True)
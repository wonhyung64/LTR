#%%
import os
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

cov2 = np.ones([4, 4]) * 1e-9
# scale_factor = 1.
# np.fill_diagonal(cov2, diagonal * scale_factor, wrap=True)

seeds = 1000
max_sampling_num = 100000
# scale_factors = np.arange(0., 1.1, 0.1)
K = range(-5, 5)
scale_factors = [2**k for k in K]

save_dir = "./data/simulation3_2"
os.makedirs(save_dir, exist_ok=True)


#%%
for scale_factor in scale_factors:
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

    scale_factor_str = "_".join(str(scale_factor).split("."))

    np.save(f"{save_dir}/is_100_{scale_factor_str}", np.array(is_seed_100), allow_pickle=True)
    np.save(f"{save_dir}/is_1000_{scale_factor_str}", np.array(is_seed_1000), allow_pickle=True)
    np.save(f"{save_dir}/is_10000_{scale_factor_str}", np.array(is_seed_10000), allow_pickle=True)
    np.save(f"{save_dir}/is_100000_{scale_factor_str}", np.array(is_seed_100000), allow_pickle=True)

#%%
result_dir = "data/simulation3_2"


#%% line graph
sample_list = ["100_", "1000_", "10000_", "100000_"]
scale_list = ["_0_03125.", "_0_0625.", "_0_125.", "_0_25.", "_0_5.", "_1.", "_2.", "_4.", "_8.", "_16."]
scale_factors = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
sample_labels = ["100", "1000", "10000", "100000"]

total_results = []
ns_results = []
for sample_str in sample_list:
    sample_files = [file for file in os.listdir(result_dir) if file.__contains__(sample_str)]
    ns_file = [file for file in os.listdir("data/simulation") if file.__contains__(sample_str.replace("_", ".")) and file.__contains__("ns")][0]
    print(ns_file)
    ns_result = np.load(f"./data/simulation/{ns_file}") 
    ns_var = np.var(ns_result)
    ns_results.append(ns_var)
    results = []

    for scale_str in scale_list:
        scale_file = [file for file in sample_files if file.__contains__(scale_str)][0]
        print(scale_file)
        scale_result = np.load(f"./{result_dir}/{scale_file}")
        scale_var = np.var(scale_result)
        results.append(scale_var)
    total_results.append(results)


#%%
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
fig.set_facecolor('white')
for num, (label, vars) in enumerate(zip(sample_labels, total_results)):
    i = num // 2
    j = num % 2
    sns.lineplot(x=scale_factors, y=ns_results[num], label="Naive", ax=axes[i][j])
    sns.lineplot(x=scale_factors, y=vars, label="Importance", ax=axes[i][j])
    axes[i][j].set_title(f"n = {label}")
    axes[i][j].set_xlabel("Scale Factor (K)")
    axes[i][j].set_ylabel("Variance of Esimates")
    axes[i][j].legend(fontsize=15)
    options = [
        axes[i][j].title,
        axes[i][j].xaxis.label,
        axes[i][j].yaxis.label,
        axes[i][j].yaxis.offsetText,
        ]
    option_list = options + axes[i][j].get_xticklabels() + axes[i][j].get_yticklabels()
    for item in option_list:
        item.set_fontsize(20)

fig.set_tight_layout(tight=True)
plt.show()

#%%
total_results = []
for i, sample_str in enumerate(sample_list):
    sample_files = [file for file in os.listdir(result_dir) if file.__contains__(sample_str)]
    results = []
    for scale_str in scale_list:
        scale_file = [file for file in sample_files if file.__contains__(scale_str)][0]
        print(scale_file)
        scale_result = np.load(f"./{result_dir}/{scale_file}")
        scale_var = np.var(scale_result)
        relative_eff = ns_results[i] / scale_var
        results.append(relative_eff)
    total_results.append(results)


#%% 
fig = plt.figure(figsize=(10, 10))
fig.set_facecolor('white')
for label, vars in zip(sample_labels, total_results):
    sns.lineplot(x=scale_factors, y=vars, label=f"n={label}")
    plt.legend()
fig.supxlabel("Scale Factor (K)")
fig.supylabel("Relative Efficiency")

fig.set_tight_layout(tight=True)


# %%

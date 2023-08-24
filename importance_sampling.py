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

        # is_seed_samples.append(np.concatenate(is_samples))

    # is_seed_samples_arr = np.stack(is_seed_samples, axis=0)
    scale_factor_str = "_".join(str(scale_factor).split("."))

    np.save(f"{save_dir}/is_100_{scale_factor_str}", np.array(is_seed_100), allow_pickle=True)
    np.save(f"{save_dir}/is_1000_{scale_factor_str}", np.array(is_seed_1000), allow_pickle=True)
    np.save(f"{save_dir}/is_10000_{scale_factor_str}", np.array(is_seed_10000), allow_pickle=True)
    np.save(f"{save_dir}/is_100000_{scale_factor_str}", np.array(is_seed_100000), allow_pickle=True)


print("NS")
print(datetime.datetime.now())
ns_seed_100 = []
ns_seed_1000 = []
ns_seed_10000 = []
ns_seed_100000 = []
for seed in range(seeds):
    """seed 반복실험"""
    print(f"seed: {seed}")
    np.random.seed(seed=seed)

    ns_samples = []
    for sampling_num in range(max_sampling_num):
        """샘플링"""
        ns_z = np.random.multivariate_normal(mu_1, cov, 1)
        ns_samples.append(ns_z)

        if sampling_num+1 == 100:
            """샘플링 개수 100개인 경우의 추정치"""
            ns_mu = np.mean([indicator_c(z) for z in np.concatenate(ns_samples)])
            ns_seed_100.append(ns_mu)

        if sampling_num+1 == 1000:
            """샘플링 개수 1000개인 경우의 추정치"""
            ns_mu = np.mean([indicator_c(z) for z in np.concatenate(ns_samples)])
            ns_seed_1000.append(ns_mu)

        if sampling_num+1 == 10000:
            """샘플링 개수 10000개인 경우의 추정치"""
            ns_mu = np.mean([indicator_c(z) for z in np.concatenate(ns_samples)])
            ns_seed_10000.append(ns_mu)

        if sampling_num+1 == 100000:
            """샘플링 개수 100000개인 경우의 추정치"""
            ns_mu = np.mean([indicator_c(z) for z in np.concatenate(ns_samples)])
            ns_seed_100000.append(ns_mu)

np.save(f"{save_dir}/ns_100", np.array(ns_seed_100), allow_pickle=True)
np.save(f"{save_dir}/ns_1000", np.array(ns_seed_1000), allow_pickle=True)
np.save(f"{save_dir}/ns_10000", np.array(ns_seed_10000), allow_pickle=True)
np.save(f"{save_dir}/ns_100000", np.array(ns_seed_100000), allow_pickle=True)

#%%
'''
print("\n반복실험: 30, 샘플수: 100")
show_result(ns_seed_100, is_seed_100, num_seed=1000)
print("\n반복실험: 30, 샘플수: 1000")
show_result(ns_seed_1000, is_seed_1000, num_seed=1000)
print("\n반복실험: 30, 샘플수: 10000")
show_result(ns_seed_10000, is_seed_10000, num_seed=1000)
print("\n반복실험: 30, 샘플수: 100000")
show_result(ns_seed_100000, is_seed_100000, num_seed=1000)


#%%
"""boxplot"""
result_list = [
    ("1000", ns_seed_1000, is_seed_1000),
    ("10000", ns_seed_10000, is_seed_10000),
    ("100000", ns_seed_100000, is_seed_100000),
    ]
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
for i, (n, ns_seed_result, is_seed_result) in enumerate(result_list):
    result_df = result2df(ns_seed_result, is_seed_result)
    sns.boxplot(data=result_df, x="sampling", y="mu", ax=axes[i])
    axes[i].set_title(f"n = {n}")
    axes[i].set_xlabel("Method")
    axes[i].set_ylabel("Estimates of mu")
    options = [
        axes[i].title,
        axes[i].xaxis.label,
        axes[i].yaxis.label,
        ]
    option_list = options + axes[i].get_xticklabels() + axes[i].get_yticklabels()
    for item in option_list:
        item.set_fontsize(20)

fig.set_tight_layout(tight=True)

plt.show()

#%%
"""likelihood ratio 평균"""
for s in range(seeds):
    print(f"\nSEED {s} Likelihood Ratio: ")
    print(np.mean([likelihood_ratio(z, mu_1, mu_2, cov, cov) for z in is_seed_samples_arr[s]]))


"""importance sampling 샘플 liklihood"""
selected_seed = 0
lrs = [likelihood_ratio(z, mu_1, mu_2, cov, cov) for z in is_seed_samples_arr[s]]
for lr in lrs: print(lr)




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
            z = np.random.multivariate_normal(mu_2, cov, 1)
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
'''
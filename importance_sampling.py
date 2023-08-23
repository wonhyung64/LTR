#%%
import numpy as np


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

cov = np.array([
    [4,.25,.25,.25],
    [.25,4,.25,.25],
    [.25,.25,4,.25],
    [.25,.25,.25,4]
    ])

seeds = 30
max_sampling_num = 10000

#%%
"""seed마다 추정한 mu"""
ns_seed_100 = []
is_seed_100 = []
ns_seed_1000 = []
is_seed_1000 = []
ns_seed_10000 = []
is_seed_10000 = []

"""각 seed마다 importance sampling으로 추출한 샘플"""
is_seed_samples = []

for seed in range(seeds):
    """seed 반복실험"""
    print(f"seed: {seed}")
    np.random.seed(seed=seed)

    ns_samples = []
    is_samples = []
    for sampling_num in range(max_sampling_num):
        """샘플링"""
        ns_z = np.random.multivariate_normal(mu_1, cov, 1)
        is_z = np.random.multivariate_normal(mu_2, cov, 1)
        ns_samples.append(ns_z)
        is_samples.append(is_z)

        if sampling_num+1 == 100:
            """샘플링 개수 100개인 경우의 추정치"""
            ns_mu = np.mean([indicator_c(z) for z in np.concatenate(ns_samples)])
            is_mu = np.mean([indicator_c(z) * likelihood_ratio(z, mu_1, mu_2, cov, cov) for z in np.concatenate(is_samples)])
            ns_seed_100.append(ns_mu)
            is_seed_100.append(is_mu)

        if sampling_num+1 == 1000:
            """샘플링 개수 1000개인 경우의 추정치"""
            ns_mu = np.mean([indicator_c(z) for z in np.concatenate(ns_samples)])
            is_mu = np.mean([indicator_c(z) * likelihood_ratio(z, mu_1, mu_2, cov, cov) for z in np.concatenate(is_samples)])
            ns_seed_1000.append(ns_mu)
            is_seed_1000.append(is_mu)

        if sampling_num+1 == 10000:
            """샘플링 개수 10000개인 경우의 추정치"""
            ns_mu = np.mean([indicator_c(z) for z in np.concatenate(ns_samples)])
            is_mu = np.mean([indicator_c(z) * likelihood_ratio(z, mu_1, mu_2, cov, cov) for z in np.concatenate(is_samples)])
            ns_seed_10000.append(ns_mu)
            is_seed_10000.append(is_mu)

    is_seed_samples.append(np.concatenate(is_samples))

is_seed_samples_arr = np.stack(is_seed_samples, axis=0)

#%%
print("\n반복실험: 10, 샘플수: 100")
show_result(ns_seed_100, is_seed_100, num_seed=10)
print("\n반복실험: 10, 샘플수: 1000")
show_result(ns_seed_1000, is_seed_1000, num_seed=10)
print("\n반복실험: 10, 샘플수: 10000")
show_result(ns_seed_10000, is_seed_10000, num_seed=10)

print("\n반복실험: 30, 샘플수: 100")
show_result(ns_seed_100, is_seed_100, num_seed=30)
print("\n반복실험: 30, 샘플수: 1000")
show_result(ns_seed_1000, is_seed_1000, num_seed=30)
print("\n반복실험: 30, 샘플수: 10000")
show_result(ns_seed_10000, is_seed_10000, num_seed=30)


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
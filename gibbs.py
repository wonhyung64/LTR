#%%
import copy
import numpy as np
from typing import List 
from scipy.stats import truncnorm


def gibbs_sampling(
    gibbs_sample_num: int,
    burnin_num: int,
    pi: np.ndarray,
    z_vec: np.ndarray,
    mu: np.ndarray,
    omega: np.ndarray,
    ) -> np.ndarray:

    print("Burn In")
    for _ in range(burnin_num):
        z_vec_copy: np.ndarray = copy.deepcopy(z_vec)
        z_vec= gibbs_1epoch(pi, z_vec_copy, mu, omega)

    print("Gibbs Sampling")
    gibbs_samples: List = []
    for _ in range(gibbs_sample_num):
        z_vec_copy: np.ndarray = copy.deepcopy(z_vec)
        z_vec= gibbs_1epoch(pi, z_vec_copy, mu, omega)
        gibbs_samples.append(z_vec)
    gibbs_z: np.ndarray = np.array(gibbs_samples)

    return gibbs_z


def gibbs_1epoch(
    pi: np.ndarray,
    z_vec: np.ndarray,
    mu: np.ndarray,
    omega: np.ndarray
    ) -> np.ndarray:

    item_j: int
    for item_j in pi:
        lower_bound: float
        upper_bound: float
        lower_bound, upper_bound = compute_bound(z_vec, item_j)

        cond_mu: float = compute_cond_mu(item_j, pi, z_vec, mu, omega)
        cond_var: float = compute_cond_var(item_j, omega)

        trn_arg_lower: float = (lower_bound - cond_mu) / cond_var
        trn_arg_upper: float = (upper_bound - cond_mu) /cond_var

        gibbs_sample: np.ndarray = truncnorm.rvs(trn_arg_lower, trn_arg_upper, size=1)
        gibbs_z_j: float = gibbs_sample[0] * cond_var + cond_mu
        z_vec[item_j] = gibbs_z_j

    return z_vec


def compute_bound(z_vec: np.ndarray, item_j: int) -> List[float]:
    lower_bound_cond: np.ndarray = (z_vec - z_vec[item_j]) < 0
    upper_bound_cond: np.ndarray = (z_vec - z_vec[item_j]) > 0

    if np.any(lower_bound_cond):
        lower_bound: float = np.max(z_vec[lower_bound_cond])
    else:
        lower_bound: float = -np.inf

    if np.any(upper_bound_cond):
        upper_bound: float = np.min(z_vec[upper_bound_cond])
    else: 
        upper_bound: float = np.inf

    return lower_bound, upper_bound


def compute_cond_mu(
    item_j: int,
    pi: np.ndarray,
    z_vec: np.ndarray,
    mu: np.ndarray,
    omega: np.ndarray,
    ) -> float:

    sum_term: List[float] = []
    omega_jj: float = omega[item_j, item_j]
    for item_k  in pi:
        if item_k == item_j:
            continue
        omega_jk: float = omega[item_j, item_k]
        sum_term.append(omega_jk * (z_vec[item_k] - mu[item_k]))
    cond_mu: float = mu[item_j] - np.sum(sum_term) / omega_jj

    return cond_mu


def compute_cond_var(item_j: int, omega: np.ndarray) -> float:
    return 1 / omega[item_j, item_j]


#%%
if __name__ == "__main__":

    """SET UP"""
    seed = 0
    gibbs_sample_num: int = 10000
    burnin_num: int = 500

    mu: np.ndarray = np.array([1.,2.,3.,4.])
    diagonal: float = 4.
    off_diagonal: float = .25
    cov:np.ndarray = np.ones([4, 4]) * off_diagonal
    np.fill_diagonal(cov, diagonal, wrap=True)
    omega: np.ndarray = np.linalg.inv(cov)

    z_vec: np.ndarray = np.array([2., 1.5, 1., .5])
    pi: np.ndarray = np.argsort(z_vec, )[::-1]


    """EXECUTE GIBBS FOR ONE PI"""
    np.random.seed(seed)
    gibbs_z: np.ndarray = gibbs_sampling(
        gibbs_sample_num,
        burnin_num,
        pi,
        z_vec,
        mu,
        omega)


    """SHOW RESULTS"""
    print()
    print(f"Mu_hat: \n{np.mean(gibbs_z, axis=0)}")
    print()
    print(f"Covariance_hat: \n{np.cov(gibbs_z.T)}")

# %%


#%%
import numpy as np


#%%
"""SET UP"""
mu = np.array([1.,2.,3.,4.])

diagonal = 4.
off_diagonal = .25
cov = np.ones([4, 4]) * off_diagonal
np.fill_diagonal(cov, diagonal, wrap=True)

pi_1 = np.array([4,3,2,1])
pi = pi_1


#%%
import copy
# item_indices = np.arange(0, len(mu))
item_indices = np.argsort(mu)

rb_sample_num = 1000
gibbs_sample_num = 1000
burnin_num = 500
omega = np.linalg.inv(cov)

"""try"""
item_j = 0

pi_sorted = copy.deepcopy(pi)
pi_sorted.sort(order=)

omega[item_j, ]
np.argsort()
type(item_indices)



from typing import List 


def compute_cond_mu(
    item_j: int,
    item_indices: np.ndarray,
    mu: np.ndarray,
    omega: np.ndarray,
    pi: np.ndarray
    ) -> float:

    sum_term: List[float] = []
    omega_jj: float = omega[item_j, item_j]
    for item_k  in item_indices:
        if item_k == item_j:
            continue
        omega_jk: float = omega[item_j, item_k]
        sum_term.append(omega_jk / omega_jj * (pi[item_k] - mu[item_k]))
    cond_mu: float = mu[item_j] - np.sum(sum_term)

    return cond_mu


def compute_cond_var(item_j: int, cov: np.ndarray) -> float:
    return cov[item_j, item_j]


def compute_trn_mu(
    rb_sample_num: int,
    cond_mu: float,
    cond_cov: float
    ) -> float:
    cond_sample: np.ndarray = np.random.normal(cond_mu, cond_cov, rb_sample_num)

    rb_mu: float = np.mean(cond_sample) 


    return rb_mu


cond_mu = compute_cond_mu(item_j, item_indices, mu, omega, pi)
cond_var = compute_cond_var(item_j, cov)



a, b = (myclip_a - cond_mu) / cond_var, (myclip_b - cond_mu) / cond_var
item_j
from scipy.stats import truncnorm
item_j
item_indices

upper_bound_cond = (pi - pi[item_j]) > 0

lower_bound_cond = (pi - pi[item_j]) < 0

(pi - 4) < 0
(pi - 4) < 0
pi - 3

def gibbs_sampling(gibbs_sample_num, burnin_num):
    total_iter_num = burnin_num + gibbs_sample_num
    pass






# %%

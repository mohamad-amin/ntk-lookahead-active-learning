import jax
import torch
import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp
from sklearn.metrics import mutual_info_score

from src.utils import gpu_util


def _project_into_probability_simplex(y, k=10):
    u = jnp.sort(y)[::-1]
    u_cumsum = jnp.cumsum(u)
    rho_helper = u + (1. / (jnp.arange(k) + 1)) * (1 - u_cumsum)
    rho = k - jnp.argmax((rho_helper > 0)[::-1])
    lmbda = (1. / rho) * (1 - u_cumsum[rho-1])
    x = jnp.maximum(jnp.zeros(1), y + lmbda)
    return x


@jax.pmap
def _p_project_into_probability_simplex(ys):
    return jax.vmap(_project_into_probability_simplex)(ys)


@jax.jit
def _jv_compute_into_probability_simplex(ys):
    return jax.vmap(_project_into_probability_simplex)(ys)


def project_into_probability_simplex_torch(probs, use_gpu=True):
    return torch.from_numpy(np.array(project_into_probability_simplex(probs, use_gpu=use_gpu)))


def project_into_probability_simplex(probs, use_gpu=True):
    """
    https://arxiv.org/pdf/1309.1541.pdf
    """
    if use_gpu:
        probs = _p_project_into_probability_simplex(gpu_util.gpu_split(probs))
        return probs.reshape(-1, *probs.shape[2:])
    else:
        return _jv_compute_into_probability_simplex(probs)


@jax.jit
def _calc_MI(x, y, bins=10):
    c_xy = jnp.histogram2d(x, y, bins)[0]
    g, p, dof, expected = jsp.stats.chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi


def calc_MI_for_pairwise_features(X):
    d = X.shape[1]
    mis = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i <= j:
                c_xy = np.histogram2d(X[:, i], X[:, j], 10)[0]
                mis[i, j] = mutual_info_score(None, None, contingency=c_xy)
            else:
                mis[i, j] = mis[j, i]
    return mis
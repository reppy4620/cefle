import jax.numpy as jnp
from abc import abstractmethod


class SDE:
    @abstractmethod
    def sde(self, x, t):
        pass

    def reverse_sde(self, score, x, t):
        drift, diffusion = self.sde(x, t)
        drift = drift - (diffusion ** 2)[:, None, None, None] * score
        return drift, diffusion

    def probability_flow(self, score, x, t):
        drift, diffusion = self.sde(x, t)
        drift = drift - 0.5 * (diffusion ** 2)[:, None, None, None] * score
        diffusion = jnp.zeros_like(diffusion)
        return drift, diffusion

    @abstractmethod
    def marginal_prob(self, x, t):
        pass


class SubVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20.):
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        _d = 1. - jnp.exp(-2. * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = jnp.sqrt(beta_t * _d)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = jnp.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1. - jnp.exp(2. * log_mean_coeff)
        return mean, std

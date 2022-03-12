import jax
import jax.numpy as jnp
import haiku as hk

from .layers import (
    GaussianFourierProjection,
    ConvNextBlock, PreNormAttention
)


class ScoreEstimator(hk.Module):
    def __init__(
        self,
        marginal_prob,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        channels=3
    ):
        super(ScoreEstimator, self).__init__()
        self.marginal_prob = marginal_prob
        self.dim = dim
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(dims[:-1], dims[1:]))
        self.mid_dim = self.in_out[-1][1]
        self.dim_mults = dim_mults
        self.channels = channels

    def __call__(self, x, t):
        emb = hk.Sequential([
            GaussianFourierProjection(self.dim, scale=16.),
            hk.Linear(self.dim * 4),
            jax.nn.gelu,
            hk.Linear(self.dim)
        ])(t)

        h = list()
        for i, (in_dim, out_dim) in enumerate(self.in_out):
            x = ConvNextBlock(in_dim, out_dim, time_emb_dim=self.dim, norm=i != 0)(x, emb)
            x = ConvNextBlock(out_dim, out_dim, time_emb_dim=self.dim)(x, emb)
            x = PreNormAttention(out_dim)(x)
            h.append(x)
            if i < len(self.in_out) - 1:
                x = hk.Conv2D(out_dim, kernel_shape=4, stride=2)(x)
        x = ConvNextBlock(self.mid_dim, self.mid_dim, time_emb_dim=self.dim)(x, emb)
        x = PreNormAttention(self.mid_dim)(x)
        x = ConvNextBlock(self.mid_dim, self.mid_dim, time_emb_dim=self.dim)(x, emb)
        for i, (in_dim, out_dim) in enumerate(reversed(self.in_out[1:])):
            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = ConvNextBlock(out_dim * 2, in_dim, time_emb_dim=self.dim, norm=i != 0)(x, emb)
            x = ConvNextBlock(in_dim, in_dim, time_emb_dim=self.dim)(x, emb)
            x = PreNormAttention(in_dim)(x)
            if i < len(self.in_out) - 1:
                x = hk.Conv2DTranspose(in_dim, kernel_shape=4, stride=2)(x)
        x = ConvNextBlock(self.dim, self.dim)(x)
        x = hk.Conv2D(self.channels, kernel_shape=1)(x)
        x = x / self.marginal_prob(x, t)[1][:, None, None, None]
        return x

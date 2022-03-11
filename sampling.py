import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

from train import build_forward_fn, State
from module import SubVPSDE


def main():
    img_shape = 32
    eps = 1e-3
    batch_size = 16

    output_dir = Path('./out/sample')
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(42)

    ckpt_dir = Path('./out')
    last_ckpt = list(sorted(ckpt_dir.glob('*.ckpt')))[-1]
    state = State.load(last_ckpt)

    sde = SubVPSDE()
    net = hk.without_apply_rng(hk.transform(build_forward_fn(sde.marginal_prob)))

    time_shape = (batch_size,)
    sample_shape = (batch_size, img_shape, img_shape, 3)
    rng, step_rng = jax.random.split(rng)
    z = jax.random.normal(step_rng, sample_shape)

    def ode_func(t, x):
        sample = jnp.asarray(x, dtype=jnp.float32).reshape(sample_shape)
        time_steps = np.ones(time_shape) * t
        score = net.apply(state.params, sample, time_steps)
        drift, diffusion = sde.probability_flow(score, sample, time_steps)
        return np.asarray(drift).reshape((-1,)).astype(np.float64)

    res = solve_ivp(ode_func, (1., eps), np.asarray(z).reshape(-1).astype(np.float64),
                    rtol=1e-5, atol=1e-5, method='RK45')
    sample = res.y[:, -1].reshape(sample_shape)
    for i in range(sample.shape[0]):
        plt.imshow(sample[i], aspect='auto')
        plt.savefig(output_dir / f'{i}.png')
        plt.close()


if __name__ == '__main__':
    main()

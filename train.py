import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import functools
import pickle

from pathlib import Path
from typing import NamedTuple
from argparse import ArgumentParser
from tqdm import tqdm

import params
from module import ScoreEstimator, SDE, SubVPSDE
from module.dataset import load_dataset, load_cifar


def build_forward_fn(marginal_prob):
    def forward_fn(x, t):
        return ScoreEstimator(marginal_prob=marginal_prob)(x, t)
    return forward_fn


def build_loss_fn(sde: SubVPSDE, net: hk.Transformed):
    def loss_fn(rng: jax.random.PRNGKey, params: hk.Params, data: jnp.ndarray):
        rng, step_rng = jax.random.split(rng)
        t = jax.random.uniform(step_rng, shape=(data.shape[0],), minval=1e-5, maxval=1.)
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, shape=data.shape)
        mean, std = sde.marginal_prob(data, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = net.apply(params, perturbed_data, t)
        loss = jnp.mean(jnp.sum((score * std[:, None, None, None] + z) ** 2, axis=(1, 2, 3)))
        return loss
    return loss_fn


class State(NamedTuple):
    step: int
    rng: jax.random.PRNGKey
    opt_state: optax.OptState
    params: hk.Params

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'step': self.step,
                'rng': self.rng,
                'opt_state': self.opt_state,
                'params': self.params
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return State(
            step=obj['step'],
            rng=obj['rng'],
            opt_state=obj['opt_state'],
            params=obj['params']
        )


class Updater:
    def __init__(self, sde, net, loss_fn, optimizer):
        self.sde: SDE = sde
        self.net: hk.Transformed = net
        self.loss_fn = loss_fn
        self.opt: optax.GradientTransformation = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, rng, data, t):
        out_rng, init_rng = jax.random.split(rng)
        params = self.net.init(init_rng, data, t)
        opt_state = self.opt.init(params)
        return State(
            step=1,
            rng=out_rng,
            opt_state=opt_state,
            params=params
        )

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: State, data: jnp.ndarray):
        rng, step_rng = jax.random.split(state.rng)
        loss, grads = jax.value_and_grad(self.loss_fn, argnums=1)(step_rng, state.params, data)
        updates, opt_state = self.opt.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        state = State(
            step=state.step + 1,
            rng=rng,
            opt_state=opt_state,
            params=params
        )
        return loss, state


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sde = SubVPSDE()
    net = hk.without_apply_rng(hk.transform(build_forward_fn(sde.marginal_prob)))
    loss_fn = build_loss_fn(sde, net)

    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=3e-4, b1=0.9, b2=0.98)
    )

    # ds = load_dataset(args.data_dir, batch_size=params.batch_size)
    ds = load_cifar(batch_size=params.batch_size)
    updater = Updater(sde, net, loss_fn, opt)

    rng = jax.random.PRNGKey(params.seed)
    data = next(ds)
    rng, step_rng = jax.random.split(rng)
    t = jax.random.uniform(step_rng, shape=(data.shape[0],))
    state = updater.init(rng, data, t)
    ckpts = list(sorted(output_dir.glob('*.ckpt')))
    if len(ckpts) > 0:
        state = state.load(ckpts[-1])
        print(f'Loaded {state.step} checkpoint')

    print('Starting training loop')
    bar = tqdm(total=params.n_steps+1 - int(state.step))
    for step in range(state.step, params.n_steps+1):
        bar.set_description_str(f'Step: {step}')
        data = next(ds)
        loss, state = updater.update(state, data)
        bar.update()
        bar.set_postfix_str(f'Loss: {loss:.06f}')

        if (step + 1) % params.save_interval == 0:
            state.save(output_dir / f'n_{step:07d}.ckpt')


if __name__ == '__main__':
    main()

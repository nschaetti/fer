"""Training script to fit CPPNs to target images via SGD."""

import os
from functools import partial
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from cppn import CPPN, FlattenCPPNParameters
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("model")
group.add_argument("--arch", type=str, default="", help="architecture")
group = parser.add_argument_group("data")
group.add_argument("--img_file", type=str, default=None, help="path of image file")

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters", type=int, default=100000, help="number of iterations")
group.add_argument("--lr", type=float, default=3e-3, help="learning rate")
group.add_argument("--init_scale", type=str, default="default", help="initialization scale")

def parse_args(*args, **kwargs):
    """Parse CLI arguments and coerce 'none' strings to `None`.

    Args:
        *args: Positional overrides passed to argparse.
        **kwargs: Keyword overrides passed to argparse.

    Returns:
        argparse.Namespace with normalized values.
    """
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
        # end if
    # end for
    return args
# end def parse_args

def main(args):
    """Train a CPPN to reproduce a reference image via SGD.

    Args:
        args: Parsed argparse namespace controlling training.
    """
    print(args)

    target_img = jnp.array(plt.imread(args.img_file)[:, :, :3])

    cppn = FlattenCPPNParameters(CPPN(args.arch, init_scale=args.init_scale))
    # cppn = FlattenCPPNParameters(CPPN(args.arch))

    rng = jax.random.PRNGKey(args.seed)
    params = cppn.init(rng)

    def loss_fn(params, target_img):
        """Compute L2 reconstruction loss for generated image.

        Args:
            params: Flattened CPPN parameters.
            target_img: Target RGB image tensor.

        Returns:
            Scalar MSE loss value.
        """
        img = cppn.generate_image(params, img_size=256)
        return jnp.mean((img - target_img)**2)
    # end def loss_fn

    @jax.jit
    def train_step(state, _):
        """Perform one optimizer update with normalized gradients.

        Args:
            state: Current `TrainState` holding params and optimizer state.
            _: Dummy loop carry (unused).

        Returns:
            Tuple of updated state and batch loss.
        """
        loss, grad = jax.value_and_grad(loss_fn)(state.params, target_img)
        grad = grad / jnp.linalg.norm(grad)
        state = state.apply_gradients(grads=grad)
        return state, loss
    # end def train_step

    tx = optax.adam(learning_rate=args.lr)
    state = TrainState.create(apply_fn=None, params=params, tx=tx)

    gen_img_fn = jax.jit(partial(cppn.generate_image, img_size=256))
    losses, imgs_train = [], [gen_img_fn(state.params)]
    pbar = tqdm(range(args.n_iters//100))
    for i_iter in pbar:
        state, loss = jax.lax.scan(train_step, state, None, length=100)
        # state, (loss, grad_norm) = jax.lax.scan(train_step, state, None, length=1)
        # print(loss, grad_norm)
        losses.append(loss)

        pbar.set_postfix(loss=loss.mean().item())
        if i_iter < 100:
            img = gen_img_fn(state.params)
            imgs_train.append(img)
        # end if
    # end for

    losses = np.array(jnp.concatenate(losses))
    imgs_train = np.array(jnp.stack(imgs_train))
    params = state.params
    img = gen_img_fn(params)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        util.save_pkl(args.save_dir, "args", args)
        util.save_pkl(args.save_dir, "arch", args.arch)
        util.save_pkl(args.save_dir, "params", params)
        plt.imsave(f"{args.save_dir}/img.png", np.array(img))

        util.save_pkl(args.save_dir, "losses", losses)
        # util.save_pkl(args.save_dir, "imgs_train", imgs_train)
    # end if
# end def main

if __name__ == '__main__':
    main(parse_args())
# end if

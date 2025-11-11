"""Composable pattern producing network definitions and helpers."""

from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
from jax.random import split
import flax
import flax.linen as nn

from einops import rearrange

import evosax

from color import hsv2rgb

cache = lambda x: x
identity = lambda x: x
cos = jnp.cos
sin = jnp.sin
tanh = jnp.tanh
sigmoid = lambda x: jax.nn.sigmoid(x) * 2. - 1.
gaussian = lambda x: jnp.exp(-x**2) * 2. - 1.
relu = jax.nn.relu
activation_fn_map = dict(cache=cache, identity=identity, cos=cos, sin=sin, tanh=tanh, sigmoid=sigmoid, gaussian=gaussian, relu=relu)

class CPPN(nn.Module):
    """Flax module implementing a CPPN with configurable activations and widths.

    Attributes:
        arch: Architecture spec as `<layers>;<act>:<width>,...`.
        inputs: Comma-separated input coordinate names.
        init_scale: `"default"` for Lecun init or float scaling factor.
    """

    arch: str = "12;cache:15,gaussian:4,identity:2,sin:1"  # default layer and neuron mix
    inputs: str = "y,x,d,b"  # "x,y,d,b,xabs,yabs"
    init_scale: str = "default"

    @nn.compact
    def __call__(self, x):
        """Run the CPPN forward pass for a batch of coordinates.

        Args:
            x: Input tensor whose last dimension matches `len(self.inputs)`.

        Returns:
            Tuple of `(h, s, v)` activations and the list of intermediate features.
        """
        n_layers, activation_neurons = self.arch.split(";")
        n_layers = int(n_layers)

        activations = [i.split(":")[0] for i in activation_neurons.split(",")]
        d_hidden = [int(i.split(":")[-1]) for i in activation_neurons.split(",")]
        dh_cumsum = list(np.cumsum(d_hidden))

        features = [x]
        for i_layer in range(n_layers):
            if self.init_scale == "default":
                x = nn.Dense(sum(d_hidden), use_bias=False)(x)
            else:
                kernel_init = nn.initializers.variance_scaling(scale=float(self.init_scale), mode="fan_in", distribution="truncated_normal")
                x = nn.Dense(sum(d_hidden), use_bias=False, kernel_init=kernel_init)(x)
            # end if
            x = jnp.split(x, dh_cumsum)
            x = [activation_fn_map[activation](xi) for xi, activation in zip(x, activations)]
            x = jnp.concatenate(x)

            features.append(x)
        # end for
        x = nn.Dense(3, use_bias=False)(x)
        features.append(x)
        # h, s, v = jax.nn.tanh(x) # CHANGED THIS TO TANH
        h, s, v = x
        return (h, s, v), features
    # end def __call__

    def generate_image(self, params, img_size=256, return_features=False):
        """Render the CPPN by sweeping a coordinate grid.

        Args:
            params: Flax parameters or flattened vector (via `FlattenCPPNParameters`).
            img_size: Spatial resolution of the square output image.
            return_features: Whether to also return intermediate activations.

        Returns:
            RGB image tensor with shape `(img_size, img_size, 3)` and optional features.
        """
        inputs = {}
        x = y = jnp.linspace(-1, 1, img_size)
        inputs['x'], inputs['y'] = jnp.meshgrid(x, y, indexing='ij')
        inputs['d'] = jnp.sqrt(inputs['x']**2 + inputs['y']**2) * 1.4
        inputs['b'] = jnp.ones_like(inputs['x'])
        inputs['xabs'], inputs['yabs'] = jnp.abs(inputs['x']), jnp.abs(inputs['y'])
        inputs = [inputs[input_name] for input_name in self.inputs.split(",")]
        inputs = jnp.stack(inputs, axis=-1)
        (h, s, v), features = jax.vmap(jax.vmap(partial(self.apply, params)))(inputs)
        r, g, b = hsv2rgb((h+1)%1, s.clip(0,1), jnp.abs(v).clip(0, 1))
        rgb = jnp.stack([r, g, b], axis=-1)
        if return_features:
            return rgb, features
        else:
            return rgb
        # end if
    # end def generate_image
# end class CPPN

class FlattenCPPNParameters():
    """Utility to flatten and unflatten CPPN parameters via EvoSax reshaper."""

    def __init__(self, cppn):
        """Initialize reshaper for a CPPN instance.

        Args:
            cppn: Instantiated `CPPN` module whose parameters to flatten.
        """
        self.cppn = cppn

        rng = jax.random.PRNGKey(0)
        d_in = len(self.cppn.inputs.split(","))
        self.param_reshaper = evosax.ParameterReshaper(self.cppn.init(rng, jnp.zeros((d_in,))))
        self.n_params = self.param_reshaper.total_params
    # end def __init__

    def init(self, rng):
        """Sample and flatten random CPPN parameters.

        Args:
            rng: JAX PRNG key used for parameter initialization.

        Returns:
            1-D array containing flattened parameters.
        """
        d_in = len(self.cppn.inputs.split(","))
        params = self.cppn.init(rng, jnp.zeros((d_in,)))
        return self.param_reshaper.flatten_single(params)
    # end def init

    def generate_image(self, params, img_size=256, return_features=False):
        """Render using flattened parameters by reshaping first.

        Args:
            params: Flattened CPPN parameter vector.
            img_size: Output image resolution.
            return_features: Whether to also return feature activations.

        Returns:
            Rendered RGB image tensor, plus optional features.
        """
        params = self.param_reshaper.reshape_single(params)
        return self.cppn.generate_image(params, img_size=img_size, return_features=return_features)
    # end def generate_image
# end class FlattenCPPNParameters

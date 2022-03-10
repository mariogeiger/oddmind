import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from e3nn_jax import Gate, Irreps, index_add
from e3nn_jax.experimental.voxel_convolution import Convolution

import nibabel as nib


def cerebellum(i):
    image = nib.load(f'x{i}.nii.gz')
    label = nib.load(f'y{i}.nii.gz')

    assert (image.affine == label.affine).all()
    assert image.header.get_zooms() == (1, 1, 1)

    image = image.get_fdata() / 600
    label = label.get_fdata()

    image = image[::2, ::2, ::2]
    label = label[::2, ::2, ::2]

    odd_label = np.zeros_like(label)

    # left cerebellum:
    odd_label[label == 6] = 1
    odd_label[label == 7] = 1
    odd_label[label == 8] = 1
    # right cerebellum:
    odd_label[label == 45] = -1
    odd_label[label == 46] = -1
    odd_label[label == 47] = -1

    return image, odd_label


# Model
@hk.without_apply_rng
@hk.transform
def model(x):
    mul0 = 35
    mul1 = 6
    gate = Gate(
        f'{mul0}x0e + {mul0}x0o', [jax.nn.gelu, jnp.tanh],
        f'{2 * mul1}x0e', [jax.nn.sigmoid], f'{mul1}x1e + {mul1}x1o'
    )

    def g(x):
        y = jax.vmap(gate)(x.reshape(-1, x.shape[-1]))
        y = y.contiguous
        return y.reshape(x.shape[:-1] + (-1,))

    kw = dict(
        irreps_sh=Irreps('0e + 1o + 2e'),
        diameter=2 * 4.3,
        num_radial_basis=3,
        steps=(1.0, 1.0, 1.0)
    )

    x = x[..., None]
    x = g(Convolution(Irreps('0e'), gate.irreps_in, **kw)(x))

    for _ in range(3):
        x = g(Convolution(gate.irreps_out, gate.irreps_in, **kw)(x))

    x = Convolution(gate.irreps_out, Irreps('0o'), **kw)(x)
    x = x[..., 0]
    return x


def main():
    # Optimizer
    learning_rate = 0.5
    opt = optax.adam(learning_rate)

    # Update function
    @jax.jit
    def update(params, opt_state, x, y):
        def loss_fn(params):
            pred = model.apply(params, x)
            absy = jnp.abs(y)
            loss = absy * jnp.log(1.0 + jnp.exp(-pred * y))
            loss = loss + (1.0 - absy) * jnp.square(pred)
            loss = jnp.mean(loss)
            return loss, pred

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, pred), grads = grad_fn(params)

        pred = pred.ravel()
        y = y.ravel()
        a = jnp.sign(jnp.round(pred)) == y
        i = jnp.asarray(y + 1, jnp.int32)
        correct = index_add(i, a, 3)
        total = index_add(i, jnp.ones_like(a), 3)
        accuracy = correct / total

        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, accuracy, pred

    x, y = cerebellum(1)
    x = x[None]
    y = y[None]

    @jax.jit
    def train(params, opt_state):
        params, opt_state, loss, accuracy, pred = update(params, opt_state, x, y)
        return params, opt_state, loss, accuracy, pred.min(), pred.max()

    np.set_printoptions(precision=2, suppress=True)

    # Init
    rng = jax.random.PRNGKey(2)
    params = model.init(rng, x)
    opt_state = opt.init(params)

    # Train
    for i in range(2000):
        params, opt_state, loss, accuracy, pred_min, pred_max = train(params, opt_state)
        print(f"[{i}] loss = {loss:.2e}  accuracy = {100 * accuracy}%  pred = [{pred_min:.2e}, {pred_max:.2e}]")


if __name__ == '__main__':
    main()

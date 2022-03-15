import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from e3nn_jax import BatchNorm, Gate, Irreps, IrrepsData, index_add
from e3nn_jax.experimental.voxel_convolution import Convolution
from e3nn_jax.experimental.voxel_pooling import zoom


# Model
@hk.without_apply_rng
@hk.transform
def model(x):
    kw = dict(
        irreps_sh=Irreps('0e + 1o + 2e'),
        diameter=5.0,
        num_radial_basis=2,
        steps=(1.0, 1.0, 1.0)
    )

    def cbg(x, mul):
        mul0, mul1, mul2 = 4 * mul, 2 * mul, mul
        gate = Gate(
            f'{mul0}x0e + {mul0}x0o', [jax.nn.gelu, jnp.tanh],
            f'{2 * mul1 + 2 * mul2}x0e', [jax.nn.sigmoid], f'{mul1}x1e + {mul1}x1o + {mul2}x2e + {mul2}x2o',
        )
        for _ in range(1 + 3):  # vectorize for axies [batch, x, y, z]
            gate = jax.vmap(gate)

        x = Convolution(x.irreps, gate.irreps_in, **kw)(x.contiguous)
        x = BatchNorm(gate.irreps_in, instance=True)(x)
        x = gate(x)
        return x

    def down(x):  # TODO replace by pool max
        def pool(x):
            return hk.avg_pool(
                x,
                window_shape=(1, 2, 2, 2, 1),
                strides=(1, 2, 2, 2, 1),
                padding='SAME',
            )
        return jax.tree_map(pool, x)

    def up(x):
        def z0(x):
            return zoom(x, 2.0)  # bilinear interpolation
        z1 = jax.vmap(z0, -1, -1)
        z2 = jax.vmap(z1, -1, -1)
        return IrrepsData(x.irreps, z1(x.contiguous), jax.tree_map(z2, x.list))

    x = x[..., None]
    x = IrrepsData.from_contiguous("0e", x)

    mul = 3

    # Block A
    x = cbg(x, mul)
    x_a = x = cbg(x, mul)
    x = down(x)

    # Block B
    x = cbg(x, 2 * mul)
    x_b = x = cbg(x, 2 * mul)
    x = down(x)

    # Block C
    x = cbg(x, 4 * mul)
    x_c = x = cbg(x, 4 * mul)
    x = down(x)

    # Block D
    x = cbg(x, 8 * mul)
    x = cbg(x, 8 * mul)

    # Block E
    x = up(x)
    x = IrrepsData.cat([x, x_c])
    x = cbg(x, 4 * mul)
    x = cbg(x, 4 * mul)

    # Block F
    x = up(x)
    x = IrrepsData.cat([x, x_b])
    x = cbg(x, 2 * mul)
    x = cbg(x, 2 * mul)

    # Block G
    x = up(x)
    x = IrrepsData.cat([x, x_a])
    x = cbg(x, mul)
    x = cbg(x, mul)  # dim = 4 * mul + 2 * 3 * mul + 5 * mul = 15 * mul

    x = Convolution(x.irreps, Irreps(f'{8 * mul}x0e'), **kw)(x.contiguous)

    for h in [8 * mul, 1]:
        x = hk.InstanceNorm(create_scale=True, create_offset=True, data_format="N...C")(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(h, with_bias=True)(x)

    x = x[..., 0]
    return x


def cerebellum(i):
    import nibabel as nib

    image = nib.load(f'x{i}.nii.gz')
    label = nib.load(f'y{i}.nii.gz')

    assert (image.affine == label.affine).all()
    assert image.header.get_zooms() == (1, 1, 1)

    image = image.get_fdata() / 600
    label = label.get_fdata()

    even_label = -np.ones_like(label)

    # brain stem
    even_label[label == 16] = 1

    # left cerebellum:
    even_label[label == 6] = 1
    even_label[label == 7] = 1
    even_label[label == 8] = 1
    # right cerebellum:
    even_label[label == 45] = 1
    even_label[label == 46] = 1
    even_label[label == 47] = 1

    return image, even_label


def main():
    wandb.init(project="oddmind")

    print('start script', flush=True)
    size = 128

    def unpad(z):
        n = 8
        return z[..., n:-n, n:-n, n:-n]

    # Optimizer
    learning_rate = 5e-3
    opt = optax.adam(learning_rate)

    # Update function
    @jax.jit
    def update(params, opt_state, x, y):
        assert x.ndim == 1 + 3
        assert y.ndim == 1 + 3

        y = unpad(y)

        def loss_fn(params):
            pred = model.apply(params, x)
            assert pred.ndim == 1 + 3
            pred = unpad(pred)
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

    np.set_printoptions(precision=2, suppress=True)

    rng = jax.random.PRNGKey(2)
    x = jnp.ones((1, size, size, size))
    print('initialize...', flush=True)
    params = model.init(rng, x)
    print('init. done', flush=True)
    opt_state = opt.init(params)

    x_data, y_data = cerebellum(1)
    x_data = x_data[None]
    y_data = y_data[None]

    def random_patch(x, y, n):
        while True:
            xi = np.random.randint(0, x.shape[1] - n + 1)
            yi = np.random.randint(0, x.shape[2] - n + 1)
            zi = np.random.randint(0, x.shape[3] - n + 1)
            x_patch = x[..., xi:xi + n, yi:yi + n, zi:zi + n]
            y_patch = y[..., xi:xi + n, yi:yi + n, zi:zi + n]
            if np.sum(unpad(y_patch) == -1) > 0 and np.sum(unpad(y_patch) == 1) > 0:
                return x_patch, y_patch

    print('start training (compiling)', flush=True)
    for i in range(2000):
        x_patch, y_patch = random_patch(x_data, y_data, size)
        params, opt_state, loss, accuracy, pred = update(params, opt_state, x_patch, y_patch)
        print(f"[{i}] loss = {loss:.2e}  accuracy = {100 * accuracy}%  pred = [{pred.min():.2e}, {pred.max():.2e}]", flush=True)
        wandb.log({'accuracy_background': accuracy[0], 'accuracy_thing': accuracy[2], 'loss': loss})


if __name__ == "__main__":
    main()

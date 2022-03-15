import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from e3nn_jax import BatchNorm, Gate, Irrep, Irreps, IrrepsData, index_add
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

    def cbg(x, mul, ir_filter=None):
        mul0, mul1, mul2 = 4 * mul, 2 * mul, mul
        irreps_scalar = Irreps(f'{mul0}x0e + {mul0}x0o')
        irreps_gated = Irreps(f'{mul1}x1e + {mul1}x1o + {mul2}x2e + {mul2}x2o')

        if ir_filter:
            irreps_scalar = irreps_scalar.filter(ir_filter)
            irreps_gated = irreps_gated.filter(ir_filter)

        gate = Gate(
            irreps_scalar, [{Irrep('0e'): jax.nn.gelu, Irrep('0o'): jnp.tanh}[ir] for _, ir in irreps_scalar],
            f'{irreps_gated.num_irreps}x0e', [jax.nn.sigmoid], irreps_gated,
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
    x = cbg(x, mul, ['0o', '1e', '2o'])  # dim = 4 * mul + 2 * 3 * mul + 5 * mul = 15 * mul

    x = Convolution(x.irreps, Irreps(f'{8 * mul}x0o'), **kw)(x.contiguous)

    for h in [8 * mul, 1]:
        x = BatchNorm(f"{x.shape[-1]}x0o", instance=True)(x).contiguous
        x = jax.nn.tanh(x)
        x = hk.Linear(h, with_bias=False)(x)

    x = x[..., 0]
    return x


def cerebellum(i):
    import nibabel as nib

    image = nib.load(f'data/x{i}.nii.gz')
    label = nib.load(f'data/y{i}.nii.gz')

    assert (image.affine == label.affine).all()
    assert image.header.get_zooms() == (1, 1, 1)

    image = image.get_fdata() / 600
    label = label.get_fdata()

    odd_label = np.zeros_like(label)

    # left cerebellum:
    odd_label[label == 6] = -1
    odd_label[label == 7] = -1
    odd_label[label == 8] = -1
    # right cerebellum:
    odd_label[label == 45] = 1
    odd_label[label == 46] = 1
    odd_label[label == 47] = 1

    return image, odd_label


def unpad(z):
    n = 8
    return z[..., n:-n, n:-n, n:-n]


def accuracy(pred, y):
    pred = pred.ravel()
    y = y.ravel()
    a = jnp.sign(jnp.round(pred)) == y
    i = jnp.asarray(y + 1, jnp.int32)
    correct = index_add(i, a, 3)
    total = index_add(i, jnp.ones_like(a), 3)
    accuracy = correct / total
    return accuracy


def loss_fn(params, x, y):
    pred = model.apply(params, x)
    assert pred.ndim == 1 + 3
    pred = unpad(pred)
    absy = jnp.abs(y)
    loss = absy * jnp.log(1.0 + jnp.exp(-pred * y))
    loss = loss + (1.0 - absy) * jnp.square(pred)
    loss = jnp.mean(loss)
    return loss, pred


def main():
    wandb.init(project="oddmind")

    print('start script', flush=True)
    size = 128

    # Optimizer
    learning_rate = 5e-3
    opt = optax.adam(learning_rate)

    # Update function
    @jax.jit
    def update(params, opt_state, x, y):
        assert x.ndim == 1 + 3
        assert y.ndim == 1 + 3

        y = unpad(y)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, pred), grads = grad_fn(params, x, y)

        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, accuracy(pred, y), pred

    @jax.jit
    def test_metrics(params, x, y):
        assert x.ndim == 1 + 3
        assert y.ndim == 1 + 3

        y = unpad(y)

        loss, pred = loss_fn(params, x, y)
        assert pred.ndim == 1 + 3
        return loss, accuracy(pred, y)

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

    x_test, y_test = cerebellum(2)
    assert x_test.shape == (256, 256, 160)
    x_test = x_test[None, 16:16+128, 42:42+128, 80-64:80+64]
    y_test = y_test[None, 16:16+128, 42:42+128, 80-64:80+64]

    print('start training (compiling)', flush=True)
    for i in range(2000):
        x_patch, y_patch = random_patch(x_data, y_data, size)
        params, opt_state, train_loss, train_accuracy, train_pred = update(params, opt_state, x_patch, y_patch)
        print(f'{i:04d} train loss: {train_loss:.2f} train accuracy: {train_accuracy}', flush=True)
        test_loss, test_accuracy = test_metrics(params, x_test, y_test)
        print(f'     test loss: {test_loss:.2f} test accuracy: {test_accuracy}', flush=True)
        wandb.log({
            'train_accuracy_left': train_accuracy[0],
            'train_accuracy_background': train_accuracy[1],
            'train_accuracy_right': train_accuracy[2],
            'train_loss': train_loss,
            'train_pred_min': np.min(train_pred),
            'train_pred_max': np.max(train_pred),
            'test_accuracy_left': test_accuracy[0],
            'test_accuracy_background': test_accuracy[1],
            'test_accuracy_right': test_accuracy[2],
            'test_loss': test_loss,
        })


if __name__ == "__main__":
    main()

import pickle
import time

import haiku as hk
import jax
import jax.numpy as jnp
import nibabel as nib
import numpy as np
import optax
from e3nn_jax import BatchNorm, Irreps, IrrepsData, Linear, gate, index_add, scalar_activation
from e3nn_jax.experimental.voxel_convolution import Convolution
from e3nn_jax.experimental.voxel_pooling import zoom

import wandb


def n_vmap(n, fun):
    for _ in range(n):
        fun = jax.vmap(fun)
    return fun


# Model
@hk.without_apply_rng
@hk.transform
def model(x):
    kw = dict(irreps_sh=Irreps("0e + 1o + 2e"), diameter=5.0, num_radial_basis=2, steps=(1.0, 1.0, 1.0))

    def cbg(x, mul, ir_filter=None):
        mul0, mul1, mul2 = round(4 * mul), round(2 * mul), round(mul)
        irreps_scalar = Irreps(f"{mul0}x0e + {mul0}x0o")
        irreps_gated = Irreps(f"{mul1}x1e + {mul1}x1o + {mul2}x2e + {mul2}x2o")

        if ir_filter:
            irreps_scalar = irreps_scalar.filter(ir_filter)
            irreps_gated = irreps_gated.filter(ir_filter)

        x = Convolution(irreps_scalar + Irreps(f"{irreps_gated.num_irreps}x0e") + irreps_gated, **kw)(x)
        x = BatchNorm(instance=True)(x)
        x = n_vmap(1 + 3, lambda x: gate(x, even_act=jax.nn.gelu, odd_act=jnp.tanh, even_gate_act=jax.nn.sigmoid))(x)
        return x

    def down(x):  # TODO replace by pool max
        def pool(x):
            ones = (1,) * (x.ndim - 4)
            return hk.avg_pool(
                x,
                window_shape=(1, 2, 2, 2) + ones,
                strides=(1, 2, 2, 2) + ones,
                padding="SAME",
            )

        return jax.tree_map(pool, x)

    # down = jax.vmap(lambda x: maxpool(x, (2, 2, 2)))

    def up(x):
        def z0(x):
            return zoom(x, 2.0)  # bilinear interpolation

        z1 = jax.vmap(z0, -1, -1)
        z2 = jax.vmap(z1, -1, -1)
        return IrrepsData(x.irreps, z1(x.contiguous), jax.tree_map(z2, x.list))

    # Convert to IrrepsData
    x = IrrepsData.from_contiguous("0e", x[..., None])

    mul = 3

    # Block A
    x = cbg(x, 2 * mul, ["0e", "1o", "2e"])
    x_a = x = cbg(x, mul, ["0e", "0o", "1e", "1o"])
    x = down(x)

    # Block B
    x = cbg(x, 2 * mul)
    x_b = x = cbg(x, 1.5 * mul)
    x = down(x)

    # Block C
    x = cbg(x, 4 * mul)
    x = cbg(x, 4 * mul)
    x_c = x = cbg(x, 4 * mul)
    x = down(x)

    # Block D
    x = cbg(x, 8 * mul)
    x = cbg(x, 8 * mul)
    x = cbg(x, 8 * mul)

    # Block E
    x = up(x)
    x = IrrepsData.cat([x, x_c])
    x = cbg(x, 4 * mul)
    x = cbg(x, 4 * mul)
    x = cbg(x, 2 * mul)

    # Block F
    x = up(x)
    x = IrrepsData.cat([x, x_b])
    x = cbg(x, 2 * mul)
    x = cbg(x, 2 * mul)
    x = cbg(x, mul)

    # Block G
    x = up(x)
    x = IrrepsData.cat([x, x_a])
    x = cbg(x, mul, ["0o", "1e", "2o"])

    x = Convolution(f"{round(8 * mul)}x0o", **kw)(x)

    for h in [round(16 * mul), round(16 * mul), 1]:
        x = BatchNorm(instance=True)(x)
        x = scalar_activation(x, [jnp.tanh])
        x = n_vmap(1 + 3, Linear(f"{h}x0o"))(x)

    return x.contiguous[..., 0]  # Back from IrrepsData to jnp.array


def cerebellum(i):
    image = nib.load(f"data/x{i}.nii.gz")
    label = nib.load(f"data/y{i}.nii.gz")

    assert (image.affine == label.affine).all()
    assert image.header.get_zooms() == (1, 1, 1)

    image = image.get_fdata() / 600
    label = label.get_fdata()

    curated_label = np.zeros_like(label)

    # left cerebellum:
    curated_label[label == 6] = -1
    curated_label[label == 7] = -1
    curated_label[label == 8] = -1
    # right cerebellum:
    curated_label[label == 45] = 1
    curated_label[label == 46] = 1
    curated_label[label == 47] = 1

    return image, curated_label


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
    p = unpad(pred)
    y = unpad(y)
    absy = jnp.abs(y)
    loss = absy * jnp.log(1.0 + jnp.exp(-p * y))
    loss = loss + (1.0 - absy) * jnp.square(p)
    loss = jnp.mean(loss)
    return loss, pred


def main():
    wandb.init(project="oddmind")

    print("start script", flush=True)
    size = 128

    # Optimizer
    learning_rate = 5e-3
    opt = optax.adam(learning_rate)

    # Update function
    @jax.jit
    def update(params, opt_state, x, y):
        assert x.ndim == 1 + 3
        assert y.ndim == 1 + 3

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, pred), grads = grad_fn(params, x, y)

        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, accuracy(pred, y), pred

    @jax.jit
    def test_metrics(params, x, y):
        assert x.ndim == 1 + 3
        assert y.ndim == 1 + 3

        loss, pred = loss_fn(params, x, y)
        assert pred.ndim == 1 + 3
        return loss, accuracy(pred, y), pred

    np.set_printoptions(precision=2, suppress=True)

    rng = jax.random.PRNGKey(2)
    x = jnp.ones((1, size, size, size))
    print("initialization...", flush=True)
    params = model.init(rng, x)
    print("initialization done", flush=True)
    opt_state = opt.init(params)

    x_train, y_train = cerebellum(1)
    x_train = x_train[None]
    y_train = y_train[None]

    def random_patch(x, y, n):
        while True:
            xi = np.random.randint(0, x.shape[1] - n + 1)
            yi = np.random.randint(0, x.shape[2] - n + 1)
            zi = np.random.randint(0, x.shape[3] - n + 1)
            x_patch = x[..., xi : xi + n, yi : yi + n, zi : zi + n]
            y_patch = y[..., xi : xi + n, yi : yi + n, zi : zi + n]
            if np.sum(unpad(y_patch) == -1) > 0 and np.sum(unpad(y_patch) == 1) > 0:
                return x_patch, y_patch

    x_test, y_test = cerebellum(2)
    assert x_test.shape == (256, 256, 160)
    x_test_patch = x_test[None, 16 : 16 + 128, 42 : 42 + 128, 80 - 64 : 80 + 64]
    y_test_patch = y_test[None, 16 : 16 + 128, 42 : 42 + 128, 80 - 64 : 80 + 64]

    x_train_patch = x_train[:, 16 : 16 + 128, 42 : 42 + 128, 80 - 64 : 80 + 64]
    y_train_patch = y_train[:, 16 : 16 + 128, 42 : 42 + 128, 80 - 64 : 80 + 64]

    print("compiling...", flush=True)
    update(params, opt_state, *random_patch(x_train, y_train, size))
    test_metrics(params, x_test_patch, y_test_patch)
    print("compiling done", flush=True)

    print("start training", flush=True)
    time0 = time.perf_counter()

    for i in range(2001):
        if i == 20:
            jax.profiler.start_trace(wandb.run.dir)

        params, opt_state, train_loss, train_accuracy, train_pred = update(
            params, opt_state, *random_patch(x_train, y_train, size)
        )
        train_loss.block_until_ready()

        print(
            f"[{i:04d}:{time.perf_counter() - time0:.1f}] train loss: {train_loss:.2f} train accuracy: {train_accuracy}",
            flush=True,
        )

        state = {
            "iteration": i,
            "_runtime": time.perf_counter() - time0,
            "train_accuracy_left": train_accuracy[0],
            "train_accuracy_background": train_accuracy[1],
            "train_accuracy_right": train_accuracy[2],
            "train_loss": train_loss,
            "train_pred_min": np.min(train_pred),
            "train_pred_max": np.max(train_pred),
        }

        if i % 10 == 0:
            test_loss, test_accuracy, test_pred = test_metrics(params, x_test_patch, y_test_patch)
            test_loss.block_until_ready()

            print(
                f"[{i:04d}:{time.perf_counter() - time0:.1f}] test loss: {test_loss:.2f} test accuracy: {test_accuracy}",
                flush=True,
            )

            state.update(
                {
                    "test_accuracy_left": test_accuracy[0],
                    "test_accuracy_background": test_accuracy[1],
                    "test_accuracy_right": test_accuracy[2],
                    "test_loss": test_loss,
                    "test_pred_min": np.min(test_pred),
                    "test_pred_max": np.max(test_pred),
                }
            )

        if i == 20:
            jax.profiler.stop_trace()

        if i % 100 == 0:
            # Save parameters and predictions (train and test)
            with open(f"{wandb.run.dir}/params.{i:04d}.pkl", "wb") as f:
                pickle.dump(params, f)

            test_pred = np.array(test_pred[0], dtype=np.float64)
            test_pred = np.sign(np.round(test_pred))
            test_pred[test_pred == -1] = 6
            test_pred[test_pred == 1] = 45

            orig = nib.load("data/y2.nii.gz")
            img = nib.Nifti1Image(unpad(test_pred), orig.affine, orig.header)
            nib.save(img, f"{wandb.run.dir}/p2.{i:04d}.nii.gz")

            orig = nib.load("data/x2.nii.gz")
            img = nib.Nifti1Image(unpad(x_test_patch)[0], orig.affine, orig.header)
            nib.save(img, f"{wandb.run.dir}/x2.nii.gz")

            _, _, train_pred = test_metrics(params, x_train_patch, y_train_patch)

            train_pred = np.array(train_pred[0], dtype=np.float64)
            train_pred = np.sign(np.round(train_pred))
            train_pred[train_pred == -1] = 6
            train_pred[train_pred == 1] = 45

            orig = nib.load("data/y1.nii.gz")
            img = nib.Nifti1Image(unpad(train_pred), orig.affine, orig.header)
            nib.save(img, f"{wandb.run.dir}/p1.{i:04d}.nii.gz")

            orig = nib.load("data/x1.nii.gz")
            img = nib.Nifti1Image(unpad(x_train_patch)[0], orig.affine, orig.header)
            nib.save(img, f"{wandb.run.dir}/x1.nii.gz")

        wandb.log(state)


if __name__ == "__main__":
    main()

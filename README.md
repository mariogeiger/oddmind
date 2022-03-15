# oddmind

The aim is to train an equivariant neural network to segment the cerebelum in a brain.
It has to distinguish between the left and right cerebellum.
To do so we output an odd scalar value for each voxel.
Zero for the background, one for the left and minus one for the right.

## Dataset

We took two brains from the [Mindboggle](https://mindboggle.info/) dataset.
The files `data/x1.nii.gz` and `data/x2.nii.gz` contain the MRI data of two brains.
The files `data/y1.nii.gz` and `data/y2.nii.gz` contain the labels of the two brains.
We use the data with index 1 for training and index 2 for testing.

# Setup

To install the dependencies

```
pip install --upgrade pip
pip install --upgrade nibabel
pip install --upgrade "jax[cpu]"  # change this to get cuda support!
pip install --upgrade dm-haiku
pip install --upgrade optax
git clone https://github.com/e3nn/e3nn-jax.git
cd e3nn-jax
python setup.py develop
```

# Execute

Make sure you execute the code on a computer with a GPU otherwise it will not even compile the code
```
# wandb login  # optional
python unet_odd.py
```

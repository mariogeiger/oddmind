![Peek 2022-03-17 00-21](https://user-images.githubusercontent.com/333780/158736611-87eae726-c7d2-4d10-868d-34e6305740a2.gif)
# oddmind: learning chiral segmentation of the brain

The aim is to train an equivariant neural network to segment the cerebellum in a brain.
It has to distinguish between the left and right cerebellum.
To do so we output an odd scalar value for each voxel.
Zero for the background, one for the left and minus one for the right.

## Dataset

We took two brains from the [Mindboggle](https://mindboggle.info/) dataset.
The files `data/x1.nii.gz` and `data/x2.nii.gz` contain the MRI data of two brains.
The files `data/y1.nii.gz` and `data/y2.nii.gz` contain the labels of the two brains.
We use the data with index 1 for training and index 2 for testing.

## Setup

This project is based on [e3nn-jax](https://github.com/e3nn/e3nn-jax).

To install the dependencies:

```
pip install --upgrade pip
pip install --upgrade nibabel
pip install --upgrade "jax[cpu]"  # change this to get cuda support!
pip install --upgrade dm-haiku
pip install --upgrade optax
pip install e3nn-jax==0.4.2  # last version tested
```

## Execute

Make sure you execute the code on a computer with a GPU otherwise it will not even compile the code
```
# wandb login  # optional
python unet_odd.py
```

## Results
Prediction of the cerebellum on a test brain (`data/x2.nii.gz`) made by an O(3)-equivariant network trained during 2000 steps (8 hours on a Tesla V100 PCIe 32GB) on a single brain (`data/x1.nii.gz`).

![Peek 2022-03-17 00-03](https://user-images.githubusercontent.com/333780/158734792-731a2861-2e6e-494c-938d-5239097d6133.gif)

## Original vs group conv
Using group convolution makes it 3x faster (on V100 gpu)

![image](https://user-images.githubusercontent.com/333780/159789326-fb24426e-4c54-47ec-b878-e8019dde9b5c.png)

![image](https://user-images.githubusercontent.com/333780/159776045-ebc16228-8254-4978-850d-ff72c720c9fa.png)

We can also see that `group conv` model spend proportionally more time on non conv op
![image](https://user-images.githubusercontent.com/333780/159776832-40560dd5-af44-4700-85ab-2746f878d2ef.png)

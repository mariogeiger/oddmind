# oddmind

The aim is to train an equivariant neural network to segment the cerebelum in a brain.
It has to distinguish between the left and right cerebellum.
To do so we output an odd scalar value for each voxel.
Zero for the background, one for the left and minus one for the right.

## Dataset

We took two brains from the [Mindboggle](https://mindboggle.info/) dataset.
The files `x1.nii.gz` and `x2.nii.gz` contain the MRI data of two brains.
The files `y1.nii.gz` and `y2.nii.gz` contain the labels of the two brains.

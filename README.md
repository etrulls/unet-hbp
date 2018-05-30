# Training framework for U-Net

A straightforward implementation of a [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) for segmentation. Developed at [CVLab@EPFL](https://cvlab.epfl.ch/) as part of our work for Task 5.6.3 of the Human Brain Project (SGA1), whose ultimate goal is to develop tools for automated processing of medical images and integrate them into [ilastik](ilastik.org). For related repositories, please see:
* https://github.com/etrulls/ilastik: ilastik fork with support for our plugin.
* https://github.com/etrulls/unet-service: service to interface ilastik with pre-trained U-Net models.

The implementation allows for 2D or 3D filters. Settings are specified via configuration files: see `config/spec.cfg` for options and other `.cfg` files for examples. Datasets are not made public as they are undergoing curation, but they should be in the future (please do inquire). Using your own data should be easy: follow the examples on `dataset.py` to format it as lists of slices (2D) or stacks (3D). This code is provided as-is, without further support.

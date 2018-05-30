from networks.unet import UNet
import torch

UNet.get_shape_combinations(200, steps=4, num_convs=1)

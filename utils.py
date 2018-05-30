import hashlib
from scipy import ndimage
from skimage import morphology
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2
import torch
from IPython import embed


def draw_anno(images, labels, filename):
    for i, (b, l) in enumerate(zip(images, labels)):
        if isinstance(b, torch.Tensor):
            b = b.numpy()
        if isinstance(l, torch.Tensor):
            l = l.numpy()
        b = ((b - b.min()) / (b.max() - b.min()) * 255).astype(np.uint8)
        b = b.transpose((1, 2, 0))
        # RGB to BGR
        b = b[:, :, ::-1]
        b = b.copy()
        for p in l:
            b = cv2.circle(b, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
        cv2.imwrite("{}{}.png".format(filename, i), b)


def hash_from_dict(my_dict, do_hash=True):
    '''
    Concatenate all strings in a dictionary to generate a hash.
    '''

    raw_str = ''
    key_list = list(my_dict.keys())
    key_list.sort()
    for key in key_list:
        val = my_dict[key]

        # Recursive call for nested dicts
        if isinstance(val, dict):
            raw_str += key + ':'
            raw_str += hash_from_dict(val, do_hash=False)
        else:
            raw_str += key + ':'
            if isinstance(val, list):
                for v in val:
                    raw_str += str(v) + '|'
            else:
                raw_str += str(val) + '|'

    # Sanity check
    if do_hash and len(raw_str) == 0:
        raise RuntimeError('Empty configuration file')

    if do_hash:
        # hashed_str = hashlib.sha256(raw_str.encode()).hexdigest()
        hashed_str = hashlib.md5(raw_str.encode()).hexdigest()
        return hashed_str, raw_str
    else:
        return raw_str


def dilate_labels(data, dilation, dims=3, value=2):
    '''
    Dilate labels to ignore boundaries.
    * dilation: tuple (inner, outer) dilation, in pixels
    * dims: 2 (2D) or 3 (3D) dilation
    * value: label for the dilated pixels
    '''

    # Let us use a single value
    if isinstance(dilation, int):
        dilation = (dilation,) * dims

    if dilation == (0, 0):
        return data.copy()
    else:
        # Rank: number of dimensions
        # Connectivity -> 1: conn-4, 2: conn-8, 3: all on for 3D filters
        filt = ndimage.generate_binary_structure(rank=dims, connectivity=1)

        inner = np.zeros(data.shape).astype(bool)
        outer = np.zeros(data.shape).astype(bool)

        # Works for 2D and 3D
        for i in range(data.shape[0]):
            inner[i] = ndimage.binary_erosion(
                data[i] > 0,
                structure=filt,
                iterations=dilation[0],
                border_value=True,
            )
            outer[i] = ndimage.binary_dilation(
                data[i] > 0,
                structure=filt,
                iterations=dilation[1],
                border_value=False,
            )

        if dilation[0] == 0:
            inner.fill(False)
        else:
            inner = (data > 0) ^ inner

        if dilation[1] == 0:
            outer.fill(False)
        else:
            outer = outer ^ (data > 0)

        dilated = data.copy()
        dilated[inner + outer] = value

        return dilated


def smooth_line(line, sigma=.5, gaussian=True):
    if isinstance(line, list):
        line = np.array(line)
    return gaussian_filter(line, sigma, mode='mirror')

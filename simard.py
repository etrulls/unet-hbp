import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import tifffile
import IPython
from time import time
from skimage.segmentation import mark_boundaries
from tqdm import tqdm
import cv2


# Initial recommendations:
# data_arg.add_argument('--elastic_alpha', type=float, default=2.0)
# data_arg.add_argument('--elastic_sigma', type=float, default=0.12)
# data_arg.add_argument('--elastic_alpha_affine', type=float, default=0.04)


def elastic_transform(stack, labels, alpha, sigma, alpha_affine, rng=None, do_3d=False, verbose=False):
    '''Elastic deformation of images as described in [Simard2003] (with modifications).
       [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    '''

    if do_3d:
        raise RuntimeError('FYI this does not work unless annotations are very consistent over z')

    if rng is None:
        rng = np.random.RandomState(None)

    if stack.dtype != 'uint8':
        raise RuntimeError('Data must be uint8')
    if labels is not None:
        if labels.dtype != 'uint8':
            raise RuntimeError('Labels must be uint8')

    single_slice = False
    if stack.ndim == 3:
        single_slice = True
        stack = stack[None, ...]
        if labels is not None:
            labels = labels[None, None, ...]
    # if stack.ndim == 4:
    #     if labels is not None:
    #         labels = np.expand_dims(labels, 1)

    # Data must be <y, x, color, z>
    stack = stack.transpose(2, 3, 1, 0)
    if labels is not None:
        labels = labels.transpose(2, 3, 1, 0)

    label_values = None
    if labels is not None:
        if stack.shape != labels.shape:
            raise RuntimeError('Data and labels do not match')
        label_values = np.unique(labels)
        if len(label_values) > 2:
            raise RuntimeError('Labels must be binary')
        # Convert to [0,1]
        labels_norm = np.zeros(labels.shape, dtype=np.float)
        labels_norm[labels == label_values[0]] = 0
        labels_norm[labels == label_values[1]] = 1

    shape_2d = (stack.shape[0], stack.shape[1], 1)
    shape_3d = (stack.shape[0], stack.shape[1], stack.shape[3])
    shape_xy = stack.shape[:2]

    # Random affine transformation:
    stack_def = np.zeros(stack.shape)
    if labels is not None:
        labels_def = np.zeros(labels.shape)

    if verbose:
        print('Warping {} 2D slices'.format(stack.shape[-1]))
    for curr_slice in range(stack.shape[-1]):
        # 2d -> every slice; 3d -> only once
        # TODO: 3d affine?
        if not do_3d or curr_slice == 0:
            # Random affine (for every slice)
            center_square = np.float32(shape_xy) // 2
            square_size = min(np.float32(shape_xy)) // 3
            pts1 = np.float32([
                center_square + square_size,
                [center_square[0] + square_size, center_square[1] - square_size],
                center_square - square_size
            ])
            pts2 = pts1 + rng.uniform(
                -alpha_affine,
                alpha_affine,
                size=pts1.shape).astype(np.float32)
            M = cv2.getAffineTransform(pts1, pts2)

        for i in range(0, stack.shape[2]):
            stack_def[:, :, i, curr_slice] = cv2.warpAffine(
                stack[:, :, i, curr_slice][..., None],
                M,
                shape_xy[::-1],
                borderMode=cv2.BORDER_REFLECT_101)
            if labels is not None:
                labels_def[:, :, i, curr_slice] = cv2.warpAffine(
                    labels_norm[:, :, i, curr_slice][..., None],
                    M,
                    shape_xy[::-1],
                    borderMode=cv2.BORDER_REFLECT_101)

    # Pixel displacements
    stack_out = np.zeros(stack.shape)
    labels_out = None
    if labels is not None:
        labels_out = np.zeros(labels.shape)

    spline_interp_order = 2
    if not do_3d:
        if verbose:
            print('Interpolating over {} 2D slices'.format(stack.shape[-1]))
        dx = gaussian_filter((rng.rand(*shape_xy) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((rng.rand(*shape_xy) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(
            np.arange(shape_2d[1]),
            np.arange(shape_2d[0]))
        indices = np.reshape(y + dy, (-1,)), np.reshape(x + dx, (-1,))

        for j in range(0, stack.shape[3]):
            for i in range(0, stack.shape[2]):
                stack_out[:, :, i, j] = map_coordinates(
                    stack_def[:, :, i, j],
                    indices,
                    order=spline_interp_order,
                    mode='reflect'
                ).reshape(shape_xy)
                if labels is not None:
                    labels_out[:, :, i, j] = map_coordinates(
                        labels_def[:, :, i, j],
                        indices,
                        order=spline_interp_order,
                        mode='reflect'
                    ).reshape(shape_xy)
    else:
        if verbose:
            print('Interpolating over 3D volume (one pass)'.format(stack.shape[-1]))
        dx = gaussian_filter((rng.rand(*shape_2d) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((rng.rand(*shape_2d) * 2 - 1), sigma) * alpha
        dz = gaussian_filter((rng.rand(*shape_2d) * 2 - 1), sigma) * alpha

        x, y, z = np.meshgrid(
            np.arange(shape_2d[1]),
            np.arange(shape_2d[0]),
            np.arange(stack.shape[-1]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

        for i in range(0, stack.shape[2]):
            stack_out[:, :, i, :] = map_coordinates(
                stack_def[:, :, i, :],
                indices,
                order=spline_interp_order,
                mode='reflect'
            ).reshape(shape_3d)
            if labels is not None:
                labels_out[:, :, i, :] = map_coordinates(
                    labels_def[:, :, i, :],
                    indices,
                    order=spline_interp_order,
                    mode='reflect'
                ).reshape(shape_3d)

    # Cast back to original dtype
    stack_out = np.clip(stack_out.round(), 0, 255).astype(stack.dtype)
    if labels is not None:
        labels_binary = labels_out > .5
        labels_out = np.zeros_like(labels)
        labels_out[labels_binary == 0] = label_values[0]
        labels_out[labels_binary == 1] = label_values[1]

    # Reshape back to the original format
    stack_out = stack_out.transpose(3, 2, 0, 1)
    if labels is not None:
        labels_out = labels_out.transpose(3, 2, 0, 1)

    # Squeeze if data is a single slice
    if single_slice:
        stack_out = stack_out[0]
        if labels is not None:
            labels_out = labels_out[0]

    # Squeeze color dim on labels
    # labels_out = labels_out[0]

    if labels is None:
        return stack_out
    else:
        return stack_out, labels_out

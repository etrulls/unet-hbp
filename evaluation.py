import numpy as np
import torch
from torch.autograd import Variable
from time import time
from IPython import embed
from tqdm import tqdm
from skimage.segmentation import mark_boundaries


def overlay_mask(stack, seg, color=(1, 1, 0)):
    if isinstance(stack, list):
        stack = np.array(stack)
    if isinstance(seg, list):
        seg = np.array(seg)
    if seg.shape != stack.shape:
        raise RuntimeError('Shapes do not match')

    s = stack.shape
    vis = np.zeros((s[0], s[-2], s[-1], 3))
    for i in range(stack.shape[0]):
        vis[i] = mark_boundaries(
            stack[i, 0, :, :],
            seg[i, 0, :, :] > 0,
            color=color)
    vis = vis.transpose(0, 3, 1, 2)
    return (vis * 255).round().clip(0, 255).astype(np.uint8)


# def overlay_detections(images, seg, gt, make_stack=False):
#     # Colors
#     col_tpos = [0, .75, 0]
#     col_fpos = [0, .5, 1]
#     col_fneg = [1, 0, 0]
# 
#     if make_stack:
#         if isinstance(images, list):
#             images = list(np.array(images).transpose((1, 0, 2, 3)))
#         if isinstance(seg, list):
#             seg = list(np.array(seg).transpose((1, 0, 2, 3)))
#         if isinstance(gt, list):
#             gt = list(np.array(gt).transpose((1, 0, 2, 3)))
#         if images[0].shape != seg[0].shape != gt[0].shape:
#             raise RuntimeError('Shapes do not match')
# 
#     r = []
#     for i, (_img, _seg, _gt) in enumerate(zip(images, seg, gt)):
#         tpos = (_gt == 1) * (_seg == 1)
#         fpos = (_gt == 0) * (_seg == 1)
#         fneg = (_gt == 1) * (_seg == 0)
# 
#         # For every slice
#         s = _img.shape
#         vis = np.zeros((s[0], s[-2], s[-1], 3))
#         for i in range(_img.shape[0]):
#             # Image input format is (M, N[, 3])
#             v = _img[i].transpose(1, 2, 0) if _img.shape[1] == 3 else _img[
#                 i][0][..., None].repeat(3, axis=2)
#             v = mark_boundaries(
#                 v,
#                 tpos[i][0],
#                 color=col_tpos,
#                 mode='thick')
#             v = mark_boundaries(
#                 v,
#                 fpos[i][0],
#                 color=col_fpos,
#                 mode='inner')
#             v = mark_boundaries(
#                 v,
#                 fneg[i][0],
#                 color=col_fneg,
#                 mode='inner')
#             vis[i] = v
#         vis = vis.transpose(0, 3, 1, 2)
# 
#         # This is very slow???
#         r.append((vis * 255).round().clip(0, 255).astype(np.uint8))
#     return r


def plot_estimate_colors(seg, gt, images=None, make_stack=False):
    # TODO make work for separate images? Or different function

    # Colors
    if images:
        col_tpos = [0, 255, 0]
    else:
        col_tpos = [255, 255, 255]
    col_tneg = [0, 0, 0]
    col_fpos = [0, 0, 255]
    col_fneg = [255, 0, 0]

    if make_stack:
        if isinstance(seg, list):
            seg = [np.array(seg).transpose((1, 0, 2, 3))]
        if isinstance(gt, list):
            gt = [np.array(gt).transpose((1, 0, 2, 3))]
        if seg[0].shape != gt[0].shape:
            raise RuntimeError('Shapes do not match')
        if images:
            if isinstance(images, list):
                images = [np.array(images).transpose((1, 0, 2, 3))]
            if images[0].shape != gt[0].shape:
                raise RuntimeError('Shapes do not match')

    r = []
    for i in range(len(gt)):
        tpos = (gt[i] == 1) * (seg[i] == 1)
        tneg = (gt[i] == 0) * (seg[i] == 0)
        fpos = (gt[i] == 0) * (seg[i] == 1)
        fneg = (gt[i] == 1) * (seg[i] == 0)

        # stack
        if seg[i].ndim == 4:
            col = np.zeros(
                (3, gt[i].shape[1], gt[i].shape[2], gt[i].shape[3]),
                dtype=np.uint8)
            for j in range(3):
                col[j][tpos[0, :, :, :]] = col_tpos[j]
                col[j][tneg[0, :, :, :]] = col_tneg[j]
                col[j][fpos[0, :, :, :]] = col_fpos[j]
                col[j][fneg[0, :, :, :]] = col_fneg[j]
        # image
        elif seg[i].ndim == 3:
            col = np.zeros(
                (3, gt[i].shape[-2], gt[i].shape[-1]),
                dtype=np.uint8)
            for j in range(3):
                col[j][tpos[0, :, :]] = col_tpos[j]
                col[j][tneg[0, :, :]] = col_tneg[j]
                col[j][fpos[0, :, :]] = col_fpos[j]
                col[j][fneg[0, :, :]] = col_fneg[j]

        if images:
            # grayscale
            r.append((col.astype(float) * .3 + images[i].repeat(3, axis=0).astype(
                float) * .7).clip(0, 255).transpose((1, 0, 2, 3)).astype(np.uint8))
        else:
            r.append(col.transpose((1, 0, 2, 3)))
    return r


def compute_metrics(seg, gt):
    # TODO make this work with multiple stacks (see e.g. plot_estimate_colors)
    if isinstance(seg, list):
        seg = np.array(seg)
    if isinstance(gt, list):
        gt = np.array(gt)
    if seg.shape != gt.shape:
        raise RuntimeError('Shapes do not match')

    # remove ignored voxels from the prediction
    ignored = (gt == 2)
    seg = seg.astype(int)
    seg[ignored] = 2

    tpos = (gt == 1) * (seg == 1)
    tneg = (gt == 0) * (seg == 0)
    fpos = (gt == 0) * (seg == 1)
    fneg = (gt == 1) * (seg == 0)

    # beware: changed the following two definitions at some point
    # really depends on what you want to look at
    acc_pos = tpos.sum() / (tpos.sum() + fneg.sum())
    acc_neg = tneg.sum() / (tneg.sum() + fpos.sum())
    acc = (acc_pos + acc_neg) / 2
    jacc = tpos.sum() / (tpos.sum() + fpos.sum() + fneg.sum())

    return {
        'acc': acc,
        'acc_pos': acc_pos,
        'acc_neg': acc_neg,
        'jacc': jacc,
        'ignored': ignored.sum() / ignored.size,
    }

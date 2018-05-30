import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from os.path import isfile
import os
from IPython import embed
from time import time
import tifffile
import pickle
from shutil import copy2
from skimage.segmentation import mark_boundaries
from tqdm import tqdm
import deepdish as dd

from networks.unet import UNet


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pickle", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--use_cpu", dest="use_cpu", action="store_true")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_seg", type=str, required=True)
    parser.add_argument("--output_vis", type=str, required=True)
    parser.set_defaults(use_cpu=False)
    params = parser.parse_args()

    # Config
    # with open(params.pickle, 'rb') as f:
    #     config = pickle.load(f)
    config = dd.io.load(params.pickle)

    # Number of threads
    torch.set_num_threads(params.threads)
    if not params.use_cpu:
        print("Running on GPU {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    config["batch_size"] = params.batch_size
    config["use_cpu"] = params.use_cpu

    # Compute input shape
    c = UNet.get_optimal_shape(
        output_shape_lower_bound=config["output_size"],
        steps=config["num_unet_steps"],
        num_convs=config["num_unet_convs"])
    input_size = [int(ci) for ci in c["input"]]
    m = np.asarray(input_size) - np.asarray(config["output_size"])
    if len(np.unique(m)) == 1:
        config["margin"] = m[0]
    else:
        raise RuntimeError("Should never be here?")

    if config["model"] == "UNet":
        print("Instantiating UNet")
        model = UNet(
            steps=config["num_unet_steps"],
            num_input_channels=np.int64(1),
            first_layer_channels=config["num_unet_filters"],
            num_classes=np.int64(2),
            num_convs=config["num_unet_convs"],
            output_size=config["output_size"],
            pooling=config["pooling"],
            activation=config["activation"],
            use_dropout=config["use_dropout"],
            use_batchnorm=config["use_batchnorm"],
            init_type=config["init_type"],
            final_unit=config["final_unit"],
        )

        # Need to overwrite this
        if model.is_3d:
            config["input_size"] = model.input_size
        else:
            config["input_size"] = [model.input_size[1], model.input_size[2]]
        config["margin"] = model.margin
        print("UNet -> Input size: {}. Output size: {}".format(
            config["input_size"], config["output_size"]))
    else:
        raise RuntimeError("Unknown model")

    # Load weights
    print("Loading model: '{}'".format(params.weights))
    model.load_state_dict(torch.load(params.weights))

    if not config["use_cpu"]:
        model.cuda()

    if model.is_3d:
        config["input_size"] = model.input_size
    else:
        config["input_size"] = [model.input_size[1], model.input_size[2]]

    # Load some data
    stack = tifffile.imread(params.input)
    stack = np.expand_dims(stack, axis=1)

    # too much
    # stack = stack[:5, :1200, :1200]
    # stack = stack[:12:, :2000, :2000]

    data, mean, std = [], [], []
    for s in stack:
        data.append(s)
        mean.append(s.mean())
        std.append(s.std())

    # pad the image
    pad = config['margin']
    if model.is_3d:
        raise RuntimeError("TODO idc right now")
    else:
        pad_dims = []
        for p in pad:
            p = int(p / 2)
            pad_dims.append([p] * 2)

    data_padded = []
    for s in data:
        data_padded.append(np.pad(s, pad_dims, 'reflect'))

    model.eval()

    # Use this to process a single batch instead
    # embed()
    # if model.is_3d:
    #     batch = torch.from_numpy(
    #         np.random.randint(
    #             0,
    #             256,
    #             size=(config["batch_size"],
    #                   1,
    #                   config["input_size"][0],
    #                   config["input_size"][1],
    #                   config["input_size"][2])
    #         )
    #     ).float()
    # else:
    #     batch = torch.from_numpy(
    #         np.random.randint(
    #             0,
    #             256,
    #             size=(config["batch_size"],
    #                   1,
    #                   config["input_size"][0],
    #                   config["input_size"][1])
    #         )
    #     ).float()
    # 
    # # LCN
    # if config['use_lcn']:
    #     for i in range(batch.shape[0]):
    #         batch[i] = (batch[i] - batch[i].mean()) / batch[i].std()
    # else:
    #     # Could use training mean/std but this is just fine
    #     batch = (batch - stack.mean()) / stack.std()
    # 
    # if config["use_cpu"]:
    #     r = model.forward(Variable(batch)).data.numpy()
    # else:
    #     r = model.forward(Variable(batch).cuda()).data.cpu().numpy()

    # Test a full stack with mirroring, accounting for boundaries etc.
    # For 2D data:
    # "images" should be a list of slices size 1xMxN, mirror-padded (config['margin']/2) on each side (x-y)
    # "mean"/"std" should be a list the same size with a single value (can be constant)
    prediction = model.inference(
        {
            "images": data_padded,
            "mean": mean,
            "std": std,
        },
        config['batch_size'],
        config['use_lcn'],
        params.use_cpu,
    )

    # Treshold
    for i in range(len(prediction)):
        if model.is_3d:
            prediction[i] = prediction[i][:, 1, :, :, :] - \
                prediction[i][:, 0, :, :, :]
        else:
            prediction[i] = np.expand_dims(
                prediction[i][0, 1, :, :] - prediction[i][0, 0, :, :],
                axis=0)
    seg = []
    for r in prediction:
        seg.append(r > 0)

    # Save
    tifffile.imsave(params.output_seg, np.array(seg).astype(np.uint8) * 255)

    # For visualization
    col_seg = [1, 0, 0]
    st_img = np.array(data)
    st_seg = np.array(seg)
    if st_img.shape != st_seg.shape:
        raise RuntimeError('Shapes do not match')

    print('Plotting...')
    bar = tqdm(total=st_img.shape[0])
    vis = []
    for j, (_img, _seg) in enumerate(zip(st_img, st_seg)):
        # For every slice
        s = _img.shape
        o = np.zeros((3, s[-2], s[-1]))
        for i in range(3):
            # Image input format is (M, N[, 3])
            v = mark_boundaries(
                _img,
                _seg,
                color=col_seg[i],
                mode='thick')
            o[i] = v
        # o = o.transpose(0, 3, 1, 2)
        vis.append((o * 255).round().clip(0, 255).astype(np.uint8))
        bar.update(1)

    tifffile.imsave(params.output_vis, np.array(vis))

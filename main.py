import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from configobj import ConfigObj
from validate import Validator
from os.path import isfile, isdir
import os
from IPython import embed
from time import time
import tifffile
import pickle
from shutil import copy2
from tensorboardX import SummaryWriter
import deepdish as dd

from networks.unet import UNet
from utils import hash_from_dict
from datasets import Data, Sampler
from tools.gpumem import Occupier


def save_state(config, state):
    ts = time()
    fn = config["output"] + "/state.h5"
    dd.io.save(fn, state)
    print("Saved state to: {0:s} [{1:.02f} s.]".format(fn, time() - ts))


def save_model(config, state, model, optimizer, label):
    # Model
    ts = time()
    fn = "{}/model-{}.pth".format(config['output'], label)
    torch.save(model.state_dict(), fn)
    print("Saved model to: {0:s} [{1:.02f} s.]".format(
        fn, time() - ts))

    # Optim
    ts = time()
    fn = "{}/optim-{}.pth".format(config['output'], label)
    torch.save(optimizer.state_dict(), fn)
    print("Saved optimizer to: {0:s} [{1:.02f} s.]".format(
        fn, time() - ts))


def train(config):
    # rng
    rng = np.random.RandomState(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    # occupy
    occ = Occupier()
    if config["occupy"]:
        occ.occupy()

    # Compute input shape
    c = UNet.get_optimal_shape(
        output_shape_lower_bound=config["output_size"],
        steps=config["num_unet_steps"],
        num_convs=config["num_unet_convs"])
    input_size = [int(ci) for ci in c["input"]]
    config['margin'] = np.asarray(input_size) - np.asarray(config["output_size"])
    # m = np.asarray(input_size) - np.asarray(config["output_size"])
    # if len(np.unique(m)) == 1:
    #     config["margin"] = m[0]
    # else:
    #     raise RuntimeError("Should never be here?")
    if len(np.unique(config["margin"])) > 1:
        raise RuntimeError("Beware: this might not work?")
    data = Data(config)

    # writer
    writer = SummaryWriter(log_dir="output/logs/" + config["force_hash"])
    board = {
        'dataset': data.loss_label,
        'loss': config['loss'],
        'writer': writer,
    }

    # Save config file, for reference
    os.system('cp {} {}/{}'.format(config["config_filename"], config[
              "output"], config["config_filename"].split('/')[-1]))
    fn = config["output"] + "/config.h5"
    print("Storing config file: '{}'".format(fn))
    dd.io.save(fn, config)

    if config["model"] == "UNet":
        print("Instantiating UNet")

        model = UNet(
            steps=config["num_unet_steps"],
            num_input_channels=data.num_channels,
            first_layer_channels=config["num_unet_filters"],
            num_classes=data.num_classes,
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
        # config["margin"] = model.margin
        print("UNet -> Input size: {}. Output size: {}".format(
            config["input_size"], config["output_size"]))
    else:
        raise RuntimeError("Unknown model")
    model.cuda()

    # Sanity check
    for j in range(len(data.train_images_optim)):
        s = data.train_images_optim[j].shape
        for i in range(len(s) - 1):
            if model.input_size[i] > s[i + 1]:
                raise RuntimeError(
                    'Input patch larger than training data '
                    '({}>{}) for dim #{}, sample #{}'.format(
                        model.input_size[i], s[
                            i + 1], i, j))
    if data.val_images_mirrored:
        for j in range(len(data.val_images_mirrored)):
            s = data.val_images_mirrored[j].shape
            for i in range(len(s) - 1):
                if model.input_size[i] > s[i + 1]:
                    raise RuntimeError(
                        'Input patch larger than validation data '
                        '({}>{}) for dim #{}, sample #{}'.format(
                            model.input_size[i], s[
                                i + 1], i, j))
    if data.test_images_mirrored:
        for j in range(len(data.test_images_mirrored)):
            s = data.test_images_mirrored[j].shape
            for i in range(len(s) -1):
                if model.input_size[i] > s[i + 1]:
                    raise RuntimeError(
                        'Input patch larger than test data '
                        '({}>{}) for dim #{}, sample #{}'.format(
                            model.input_size[i], s[
                                i + 1], i, j))

    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
        )
    elif config["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
        )
    else:
        raise RuntimeError("Unsupported optimizer")

    # Load state
    first_batch = 0
    fn = config["output"] + "/state.h5"
    if isfile(fn):
        # print("Loading state: '{}'".format(fn))
        # with open(fn, "rb") as handle:
        #     state = pickle.load(handle)
        state = dd.io.load(fn)
        first_batch = state["cur_batch"] + 1
    else:
        state = {}

    # Load model
    fn = "{}/model-last.pth".format(config["output"])
    if isfile(fn):
        print("Loading model: '{}'".format(fn))
        model.load_state_dict(torch.load(fn))
    else:
        print("No model to load")

    # Load optimizer
    fn = "{}/optim-last.pth".format(config["output"])
    if isfile(fn):
        optimizer.load_state_dict(torch.load(fn))
    else:
        print("No optimizer to load")

    state.setdefault("epoch", 0)
    state.setdefault("cur_batch", 0)
    state.setdefault("loss", np.zeros(config["max_steps"]))
    state.setdefault("res_train", {"batch": [], "metrics": []})
    for t in config["test_thresholds"]:
        state.setdefault("res_train_th_{}".format(t), {"batch": [], "metrics": []})
        state.setdefault("res_val_th_{}".format(t), {"batch": [], "metrics": []})
        state.setdefault("res_test_th_{}".format(t), {"batch": [], "metrics": []})

    # TODO Learn to sample and update this accordingly
    if config["loss"] == "classification":
        # loss_criterion = torch.nn.NLLLoss(data.weights.cuda(), reduce=False)
        loss_criterion = F.nll_loss
    elif config["loss"] == "regression":
        raise RuntimeError("TODO")
    elif config['loss'] == 'jaccard' or config['loss'] == 'dice':
        from loss import OverlapLoss
        loss_criterion = OverlapLoss(
            config['loss'],
            config['overlap_loss_smoothness'],
            config['overlap_fp_factor'])
    else:
        raise RuntimeError("TODO")

    if model.is_3d:
        batch = torch.Tensor(
            config["batch_size"],
            data.num_channels,
            config["input_size"][0],
            config["input_size"][1],
            config["input_size"][2],
        )
        # labels = torch.ByteTensor(
        if not data.dot_annotations:
            labels = torch.LongTensor(
                config["batch_size"],
                config["output_size"][0],
                config["output_size"][1],
                config["output_size"][2],
            )
        else:
            labels = []
    else:
        batch = torch.Tensor(
            config["batch_size"],
            data.num_channels,
            config["input_size"][0],
            config["input_size"][1],
        )
        # labels = torch.ByteTensor(
        if not data.dot_annotations:
            labels = torch.LongTensor(
                config["batch_size"],
                config["output_size"][0],
                config["output_size"][1],
            )
        else:
            labels = []

    do_save_state = False
    model.train()

    # Sampler
    print("Instantiating sampler")
    sampler = Sampler(
        model.is_3d,
        {"images": data.train_images_optim,
         "labels": data.train_labels_optim,
         "mean": data.train_mean,
         "std": data.train_std},
        config,
        rng,
        data.dot_annotations,
    )

    if occ.is_busy():
        occ.free()

    # Loop
    for state["cur_batch"] in range(first_batch, config["max_steps"]):
        # Sample
        ts = time()
        coords = []
        elastic = []
        for i in range(config["batch_size"]):
            b, l, cur_coords, cur_elastic = sampler.sample()
            batch[i] = torch.from_numpy(b)
            if not data.dot_annotations:
                labels[i] = torch.from_numpy(l)
            else:
                labels.append(torch.from_numpy(l))
            coords.append(cur_coords)
            elastic.append(cur_elastic)

        # Forward pass
        inputs = Variable(batch).cuda()
        outputs = model(inputs)
        optimizer.zero_grad()
        if config['loss'] == 'jaccard' or config['loss'] == 'dice':
            targets = Variable(labels.float()).cuda()
            o = F.softmax(outputs, dim=1)[:, 1, :, :]
            loss = loss_criterion.forward(o, targets)
            loss = sum(loss) / len(loss)
        elif config['loss'] == 'classification':
            targets = Variable(labels).cuda()
            if data.is_3d:
                # Do it slice by slice. Ugly but it works!
                loss = []
                for z in range(outputs.shape[2]):
                    loss.append(loss_criterion(
                        F.log_softmax(outputs[:, :, z, :, :], dim=1),
                        targets[:, z, :, :],
                        weight=data.weights.cuda(),
                        reduce=True,
                        ignore_index=2))
                loss = sum(loss) / len(loss)
            else:
                # f(reduce=True) is equivalent to f(reduce=False).mean()
                # no need to average over the batch size then
                loss = loss_criterion(
                    F.log_softmax(outputs, dim=1),
                    targets,
                    weight=data.weights.cuda(),
                    reduce=True,
                    ignore_index=2)
        else:
            raise RuntimeError('Bad loss type')

        # Sanity check
        # if not data.dot_annotations and loss.data.cpu().sum() > 10:
        #     print("very high loss?")
        #     embed()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Get class stats
        ws = [0, 0]
        for l in labels:
            ws[0] += (l == 0).sum()
            ws[1] += (l == 1).sum()

        # Update state
        cur_loss = loss.data.cpu().sum()
        state["loss"][state["cur_batch"]] = cur_loss
        board['writer'].add_scalar(
            board['dataset'] + '-loss-' + board['loss'],
            cur_loss,
            state['cur_batch'])

        print("Batch {it:d} -> Avg. loss {loss:.05f}: [{t:.02f} s.] (Range: {rg:.1f})".format(
            it=state["cur_batch"] + 1,
            loss=cur_loss,
            t=time() - ts,
            rg=outputs.data.max() - outputs.data.min(),
        ))

        # Cross-validation
        force_eval = False
        if config["check_val_every"] > 0 and data.evaluate_val:
            if (state["cur_batch"] + 1) % config["check_val_every"] == 0:
                res = model.inference(
                    {
                        "images": data.val_images_mirrored,
                        "mean": data.val_mean,
                        "std": data.val_std,
                    },
                    config['batch_size'],
                    config['use_lcn'],
                )

                is_best = model.validation_by_classification(
                    images=data.val_images,
                    gt=data.val_labels_th,
                    prediction=res,
                    state=state,
                    board=board,
                    output_folder=config['output'],
                    xval_metric=config['xval_metric'],
                    dilation_thresholds=config['test_thresholds'],
                    subset='val',
                    make_stack=data.plot_make_stack,
                    force_save=False,
                )

                # Save models if they are the best at any test threshold
                for k, v in is_best.items():
                    if v is True:
                        save_model(config, state, model, optimizer, 'best_th_{}'.format(k))
                        do_save_state = True

                # Force testing on train/test
                force_eval = any(is_best.keys())

        # Test on the training data
        if config["check_train_every"] > 0 and data.evaluate_train:
            if ((state["cur_batch"] + 1) % config["check_train_every"] == 0) or force_eval:
                res = model.inference(
                    {
                        "images": data.train_images_mirrored[:data.num_train_orig],
                        "mean": data.train_mean,
                        "std": data.train_std,
                    },
                    config['batch_size'],
                    config['use_lcn'],
                )

                model.validation_by_classification(
                    images=data.train_images[:data.num_train_orig],
                    gt=data.train_labels_th,
                    prediction=res,
                    state=state,
                    board=board,
                    output_folder=config['output'],
                    xval_metric=config['xval_metric'],
                    dilation_thresholds=config['test_thresholds'],
                    subset='train',
                    make_stack=data.plot_make_stack,
                    force_save=force_eval,
                )

        # Test on the test data
        if config["check_test_every"] > 0 and data.evaluate_test:
            if ((state["cur_batch"] + 1) % config["check_test_every"] == 0) or force_eval:
                res = model.inference(
                    {
                        "images": data.test_images_mirrored,
                        "mean": data.test_mean,
                        "std": data.test_std,
                    },
                    config['batch_size'],
                    config['use_lcn'],
                )

                model.validation_by_classification(
                    images=data.test_images,
                    gt=data.test_labels_th,
                    prediction=res,
                    state=state,
                    board=board,
                    output_folder=config['output'],
                    xval_metric=config['xval_metric'],
                    dilation_thresholds=config['test_thresholds'],
                    subset='test',
                    make_stack=data.plot_make_stack,
                    force_save=force_eval,
                )

        # Also save models periodically, to resume executions
        if config["save_models_every"] > 0:
            if (state["cur_batch"] + 1) % config["save_models_every"] == 0:
                save_model(config, state, model, optimizer, 'last')
                do_save_state = True

        # Save training state periodically (or if forced)
        if do_save_state:
            save_state(config, state)
            do_save_state = False

    board['writer'].close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps-plot", type=int, default=50)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--occupy", dest="occupy", action="store_true")
    parser.set_defaults(occupy=False)
    params = parser.parse_args()

    # Number of threads
    torch.set_num_threads(params.threads)
    print("Running on GPU {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    config = ConfigObj(params.config, configspec="config/spec.cfg")
    if not config:
        raise RuntimeError(
            "Could not load the configuration file: '{}'".format(params.config))

    validator = Validator()
    ret = config.validate(validator)
    if ret is not True:
        for k in ret:
            if not ret[k]:
                print("--- {} is {}".format(k, str(ret[k])))
        raise RuntimeError(
            "Errors validating the training configuration file")

    if config["augment_elastic"] == "simple":
        raise RuntimeError("Deprecated")

    if not config["force_hash"]:
        hashname, _ = hash_from_dict(config)
        config["force_hash"] = hashname
        config["output"] = "output/torch/" + hashname
    else:
        config["output"] = "output/torch/" + config["force_hash"]

    if not isdir(config["output"]):
        os.makedirs(config["output"])
    print("Output folder: {}".format(config["output"]))

    config["config_filename"] = params.config
    config["batch_size"] = params.batch_size
    # config["steps_loss"] = params.steps_loss
    config["steps_plot"] = params.steps_plot
    if params.lr is not None:
        config["learning_rate"] = params.lr
        print("Overwriting learning rate: {}".format(config["learning_rate"]))
    if params.momentum is not None:
        config["momentum"] = params.momentum
        print("Overwriting SGD momentum: {}".format(config["momentum"]))
    config["occupy"] = params.occupy

    train(config)

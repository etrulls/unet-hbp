import os
import numpy as np
import tifffile
import nrrd
import h5py
from scipy import ndimage
from skimage import morphology
from time import time
import IPython
import torch
from torch.autograd import Variable
import utils
from networks.unet import UNet
from scipy.signal import convolve2d
from scipy.misc import imresize
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.filters import gaussian_filter as gaussian
from simard import elastic_transform
from evaluation import overlay_mask
from tqdm import tqdm
import deepdish as dd
from IPython import embed


class Data(object):
    def __init__(self, config):
        self.config = config
        self.name = config['dataset']

        ts = time()
        print('Loading dataset "{}"'.format(self.name))

        # Hippocampus
        if self.name == 'hipp-mito' or self.name == 'hipp-syn':
            self.retrieve_graham()
        # GAD mouse
        elif self.name == 'hbp2-ch1' or self.name == 'hbp2-ch2':
            self.retrieve_gad_mouse()
        # GAD mouse, extended
        elif self.name == 'hbp3-ch1' or self.name == 'hbp3-ch2' or self.name == 'hbp3-ch2-bbp':
            self.retrieve_gad_mouse_extended()
        elif self.name == 'defelipe-mito' or self.name == 'defelipe-syn':
            self.retrieve_defelipe(self.name.split('-')[-1])
        else:
            raise RuntimeError('Unknown dataset')

        # Mirror training data
        # Required at the very least for evaluation on the training data
        # Can ignore augmented copies if we only need mirroring for training,
        # but not testing
        print('Padding training data...')
        if self.config['mirror_data']:
            self.train_images_mirrored, self.train_labels_mirrored = self.mirror(
                [m // 2 for m in self.config['margin']], self.train_images, self.train_labels)
        else:
            self.train_images_mirrored, self.train_labels_mirrored = self.mirror(
                [m // 2 for m in self.config['margin']], self.train_images[
                    :self.num_train_orig], self.train_labels[
                    :self.num_train_orig])

        # Mirror validation data
        if self.val_images:
            print('Padding validation data...')
            self.val_images_mirrored, self.val_labels_mirrored = \
                self.mirror([m // 2 for m in self.config['margin']], self.val_images, self.val_labels)
        else:
            self.val_images_mirrored = None

        # Mirror test data
        if self.test_images:
            print('Padding test data...')
            self.test_images_mirrored, self.test_labels_mirrored = \
                self.mirror([m // 2 for m in self.config['margin']], self.test_images, self.test_labels)
        else:
            self.test_images_mirrored = None

        # Select data used for training, as per config
        if self.config['train_mirroring']:
            self.train_images_optim = self.train_images_mirrored
            self.train_labels_optim = self.train_labels_mirrored
        else:
            self.train_images_optim = self.train_images
            self.train_labels_optim = self.train_labels

        # TODO add regression loss here
        if self.dot_annotations:
            self.num_classes = 1
        else:
            self.num_classes = 2

            # Sanity-check the labels
            if np.unique(np.array([np.unique(s)
                                   for s in self.train_labels])).size != 2:
                raise RuntimeError('More than 2 values in the training labels?')
            if self.val_labels:
                if np.unique(np.array([np.unique(s)
                                       for s in self.val_labels])).size != 2:
                    raise RuntimeError('More than 2 values in the val labels?')
            if self.test_labels:
                if np.unique(np.array([np.unique(s)
                                       for s in self.test_labels])).size != 2:
                    raise RuntimeError('More than 2 values in the test labels?')

            # Dilate training labels
            if self.config['dilate_train_gt'] > 0:
                print('Dilating training labels (for training: {})...'.format(
                    len(self.train_labels)))
                for i in range(len(self.train_labels)):
                    self.train_labels[i] = utils.dilate_labels(
                        data=self.train_labels[i],
                        dilation=self.config['dilate_train_gt'],
                        dims=3 if self.is_3d else 2,
                        value=2)

            # Dilate training labels (with multiple testing thresholds)
            self.train_labels_th = {}
            if self.config['check_train_every'] > 0:
                print('Dilating training labels (for testing: {})...'.format(
                    self.num_train_orig))
                for t in self.config['test_thresholds']:
                    x = []
                    how_many = len(self.train_labels) if self.is_3d else \
                        self.num_train_orig
                    for i in range(how_many):
                        x.append(
                            utils.dilate_labels(
                                data=self.train_labels[i],
                                dilation=t,
                                dims=3 if self.is_3d else 2,
                                value=2)
                        )
                    self.train_labels_th[t] = x

            # Dilate validation labels
            self.val_labels_th = {}
            if self.val_images and self.config['check_val_every'] > 0:
                print('Dilating validation labels...')
                for t in self.config['test_thresholds']:
                    x = []
                    for i in range(len(self.val_labels)):
                        x.append(
                            utils.dilate_labels(
                                data=self.val_labels[i],
                                dilation=t,
                                dims=3 if self.is_3d else 2,
                                value=2)
                        )
                    self.val_labels_th[t] = x

            # Dilate test labels
            self.test_labels_th = {}
            if self.test_images and self.config['check_test_every'] > 0:
                print('Dilating test labels...')
                for t in self.config['test_thresholds']:
                    x = []
                    for i in range(len(self.test_labels)):
                        x.append(
                            utils.dilate_labels(
                                data=self.test_labels[i],
                                dilation=t,
                                dims=3 if self.is_3d else 2,
                                value=2)
                        )
                    self.test_labels_th[t] = x

            num_fg = (self.train_labels[0] == 1).sum()
            num_bg = (self.train_labels[0] == 0).sum()
            self.weights = torch.Tensor([
                num_fg / (num_fg + num_bg),
                num_bg / (num_fg + num_bg),
            ])
            print('Class weights: bg = {0:.3f}, fg = {1:.3f}'.format(
                self.weights[0], self.weights[1]))

            # Set weight to 0 to ignore
            # if self.config['dilate_train_gt']:
            #     self.num_classes += 1
            #     self.weights.resize_(self.weights.numel() + 1)
            #     self.weights[-1] = 0

        print('Loaded dataset "{d}" [{t:.2f} s.]'.format(d=self.name, t=time() - ts))

    def fill_holes(self, v, label='---'):
        '''
        Fill holes in the annotations. Input must be binary.
        '''

        if len(np.unique(v)) > 2:
            raise RuntimeError('Labels must be binary for hole filling')

        do_255 = False
        if v.max() == 255:
            v[v > 0] = 1
            do_255 = True

        t = time()
        n = v.sum()
        for i in range(len(v)):
            if len(v[i].shape) == 3 and v[i].shape[0] == 1:
                v[i][0] = binary_fill_holes(v[i][0]).astype(v.dtype)
            elif len(v[i].shape) == 2:
                v[i] = binary_fill_holes(v[i]).astype(v.dtype)
            else:
                raise RuntimeError('Should not be here')
        n_ = v.sum()
        if n != n_:
            print('Filled holes ({0:s}): {1:d} -> {2:d} (+{3:d}) [{4:.02f} s.]'.format(
                label, n, n_, n_ - n, time() - t))

        if do_255:
            v[v > 0] = 255
        return v

    def retrieve_graham(self):
        src = '/cvlabdata1/cvlab/datasets_eduard/em/tif'
        self.num_channels = 1
        self.dot_annotations = False
        self.evaluate_train = True
        self.evaluate_val = True
        self.evaluate_test = False
        self.loss_label = self.name

        if len(self.config['output_size']) == 2:
            self.is_3d = False
        elif len(self.config['output_size']) == 3:
            self.is_3d = True
        else:
            raise RuntimeError('Patch dimensions are not 2 or 3?')

        if self.is_3d:
            self.plot_make_stack = False
        else:
            self.plot_make_stack = True

        # Training data
        self.train_images = tifffile.imread(src + '/hipp-train-data.tif')
        self.train_images = np.expand_dims(self.train_images, axis=1)

        # Validation data
        self.val_images = tifffile.imread(src + '/hipp-test-data.tif')
        self.val_images = np.expand_dims(self.val_images, axis=1)

        # Test data
        self.test_images = None
        self.test_labels = None

        # Labels
        if self.name == 'hipp-syn':
            self.train_labels = tifffile.imread(src + '/hipp-train-syn.tif')
            self.val_labels = tifffile.imread(src + '/hipp-test-syn.tif')
        else:
            self.train_labels = tifffile.imread(src + '/hipp-train-mito.tif')
            self.val_labels = tifffile.imread(src + '/hipp-test-mito.tif')
        self.train_labels = np.expand_dims(self.train_labels, axis=1)
        self.val_labels = np.expand_dims(self.val_labels, axis=1)

        print('Training data: {}'.format(self.train_images.shape))
        print('Validation data: {}'.format(self.val_images.shape))

        # Relabel
        self.train_labels[self.train_labels == 255] = 1
        self.val_labels[self.val_labels == 255] = 1

        # Store original number of slices, for testing
        if self.is_3d:
            self.num_train_orig = 1
        else:
            self.num_train_orig = len(self.train_images)

        # Offline data augmentation for biomedical data
        sim_data, sim_labels = [], []
        if self.config['augment_elastic'] == 'simard':
            raise RuntimeError('TODO')

        # 2D, so make into lists
        if not self.is_3d:
            self.train_images = [s for s in self.train_images]
            self.val_images = [s for s in self.val_images]
            self.train_labels = [s for s in self.train_labels]
            self.val_labels = [s for s in self.val_labels]
            for x in sim_data:
                for s in x:
                    self.train_images.append(s)
            for x in sim_labels:
                for s in x:
                    self.train_labels.append(s)
        # 3D, so we have to flip Nx1 into 1xN
        else:
            self.train_images = [self.train_images.transpose((1, 0, 2, 3))]
            self.val_images = [self.val_images.transpose((1, 0, 2, 3))]
            self.train_labels = [self.train_labels.transpose((1, 0, 2, 3))]
            self.val_labels = [self.val_labels.transpose((1, 0, 2, 3))]
            for x, y in zip(sim_data, sim_labels):
                self.train_images.append(x.transpose((1, 0, 2, 3)))
                self.train_labels.append(y.transpose((1, 0, 2, 3)))

        # Data has a very strong bias over the z axis?
        # Compute stats, by slice or by stack (stack: probably not ideal)
        self.train_mean = [s.mean() for s in self.train_images]
        self.train_std = [s.std() for s in self.train_images]
        self.val_mean = [s.mean() for s in self.val_images]
        self.val_std = [s.std() for s in self.val_images]

    def retrieve_defelipe(self, cells):
        src = '/cvlabdata1/cvlab/datasets_eduard/defelipe/MAR'
        self.num_channels = 1
        self.dot_annotations = False
        self.evaluate_train = True
        self.evaluate_val = True
        self.evaluate_test = False
        self.loss_label = self.name

        if len(self.config['output_size']) == 2:
            self.is_3d = False
        elif len(self.config['output_size']) == 3:
            self.is_3d = True
        else:
            raise RuntimeError('Patch dimensions are not 2 or 3?')

        if self.is_3d:
            self.plot_make_stack = False
        else:
            self.plot_make_stack = True

        if cells == 'mito':
            cell_id = 1
        elif cells == 'syn':
            cell_id = 2
        else:
            raise RuntimeError('Unknown label type')

        # Training data
        self.train_images = tifffile.imread(src + '/training.tiff')
        self.train_labels = tifffile.imread(src + '/training_gt.tiff')

        # Validation data
        self.val_images = tifffile.imread(src + '/testing.tiff')
        self.val_labels = tifffile.imread(src + '/testing_gt.tiff')

        # Test data
        self.test_images = None
        self.test_labels = None

        # Labels
        print('Training data: {}'.format(self.train_images.shape))
        print('Validation data: {}'.format(self.val_images.shape))
        if self.test_images:
            print('Test data: {}'.format(self.test_images.shape))

        # Relabel
        self.train_labels = (self.train_labels == cell_id).astype(np.uint8)
        self.val_labels = (self.val_labels == cell_id).astype(np.uint8)

        # Store original number of slices, for testing
        self.num_train_orig = 1

        # Offline data augmentation for biomedical data
        sim_data, sim_labels = [], []
        # if self.config['augment_elastic'] == 'simard':
        if self.config['augment_elastic'] == 'simard':
            src_sim = '{}/simard-{}-{}-{}'.format(
                src,
                self.config['elastic_params'][1],
                self.config['elastic_params'][2],
                self.config['elastic_params'][3])
            if not os.path.isdir(src_sim):
                os.mkdir(src_sim)

            if self.name == 'defelipe-syn':
                root_data = src_sim + '/hipp-syn-data'
                root_labels = src_sim + '/hipp-syn-labels'
                root_vis = src_sim + '/hipp-syn-vis'
            elif self.name == 'defelipe-mito':
                root_data = src_sim + '/hipp-mito-data'
                root_labels = src_sim + '/hipp-mito-labels'
                root_vis = src_sim + '/hipp-mito-vis'
            else:
                raise RuntimeError('Unknown dataset')

            print('Data augmentation (Simard): {} copies'.format(
                int(self.config['elastic_params'][0])))
            for i in tqdm(range(int(self.config['elastic_params'][0]))):
                fn_data = '{0:s}-{1:05d}.tif'.format(root_data, i)
                fn_labels = '{0:s}-{1:05d}.tif'.format(root_labels, i)
                fn_vis = '{0:s}-{1:05d}.tif'.format(root_vis, i)
                if os.path.isfile(fn_data) and os.path.isfile(fn_labels):
                    sim_data.append(tifffile.imread(fn_data))
                    sim_labels.append(tifffile.imread(fn_labels))
                else:
                    sim_data_i, sim_labels_i = elastic_transform(
                        self.train_images,
                        self.train_labels,
                        alpha=self.config['elastic_params'][1],
                        sigma=self.config['elastic_params'][2],
                        alpha_affine=self.config['elastic_params'][3],
                        do_3d=self.is_3d)
                    sim_data.append(sim_data_i)
                    sim_labels.append(sim_labels_i)
                    tifffile.imsave(fn_data, sim_data_i)
                    tifffile.imsave(fn_labels, sim_labels_i)
                    # This is nice but veeeeery slow
                    # tifffile.imsave(fn_vis, overlay_mask(sim_data_i, sim_labels_i))

        # 2D, so make into lists
        if not self.is_3d:
            self.train_images = [s for s in self.train_images]
            self.val_images = [s for s in self.val_images]
            self.train_labels = [s for s in self.train_labels]
            self.val_labels = [s for s in self.val_labels]
            for x in sim_data:
                for s in x:
                    self.train_images.append(s)
            for x in sim_labels:
                for s in x:
                    self.train_labels.append(s)
        # 3D, so we have to flip Nx1 into 1xN
        else:
            self.train_images = [self.train_images.transpose((1, 0, 2, 3))]
            self.val_images = [self.val_images.transpose((1, 0, 2, 3))]
            self.train_labels = [self.train_labels.transpose((1, 0, 2, 3))]
            self.val_labels = [self.val_labels.transpose((1, 0, 2, 3))]
            for x, y in zip(sim_data, sim_labels):
                self.train_images.append(x.transpose((1, 0, 2, 3)))
                self.train_labels.append(y.transpose((1, 0, 2, 3)))

        # Do the same as for Graham's data, for consistency
        self.train_mean = [s.mean() for s in self.train_images]
        self.train_std = [s.std() for s in self.train_images]
        self.val_mean = [s.mean() for s in self.val_images]
        self.val_std = [s.std() for s in self.val_images]

    def retrieve_gad_mouse(self):
        raise RuntimeError("Deprecated")

        src = '/cvlabdata1/cvlab/datasets_eduard/hbp-catherine2'
        self.num_channels = 1
        self.is_3d = False
        self.dot_annotations = False
        self.evaluate_train = True
        self.evaluate_val = True
        self.evaluate_test = False
        self.plot_make_stack = True
        self.loss_label = self.name

        # Load data
        if self.name == 'hbp2-ch1':
            self.images = tifffile.imread(src + '/images-ch1.tif')
            self.labels = tifffile.imread(src + '/labels-ch1.tif')
            self.valid = h5py.File(src + '/valid-ch1.h5', 'r')['valid'].value
            # self.lims = [1036, 1050]
            self.lims = [1036, 950]
        elif self.name == 'hbp2-ch2':
            self.images = tifffile.imread(src + '/images-ch2.tif')
            self.labels = tifffile.imread(src + '/labels-ch2.tif')
            self.valid = h5py.File(src + '/valid-ch2.h5', 'r')['valid'].value
            # self.lims = [989, 1153]
            self.lims = [900, 1050]
        else:
            raise RuntimeError('Error')

        # Filter
        self.images = self.images[self.valid, :self.lims[0], :self.lims[1]]
        self.labels = self.labels[self.valid, :self.lims[0], :self.lims[1]]
        self.images = np.expand_dims(self.images, axis=1)
        self.labels = np.expand_dims(self.labels, axis=1)

        # Fill holes in labeling
        self.labels = self.fill_holes(self.labels, label='all')

        # Write for visualization
        fn = src + ('/images-ch1-cropped.tif' if self.name ==
                    'hbp2-ch1' else '/images-ch2-cropped.tif')
        if not os.path.isfile(fn):
            tifffile.imsave(fn, self.images)
        fn = src + ('/labels-ch1-cropped.tif' if self.name ==
                    'hbp2-ch1' else '/labels-ch2-cropped.tif')
        if not os.path.isfile(fn):
            tifffile.imsave(fn, (self.labels > 1).astype(np.uint8) * 255)
        fn = src + ('/overlay-ch1-cropped.tif' if self.name ==
                    'hbp2-ch1' else '/overlay-ch2-cropped.tif')
        if not os.path.isfile(fn):
            tifffile.imsave(fn, overlay_mask(self.images, self.labels))

        # Split into train/test
        split = [75, 25]
        fn = src + '/split-{}-{}-{}.h5'.format(
            'ch1' if self.name == 'hbp2-ch1' else 'ch2',
            split[0],
            split[1])
        if os.path.isfile(fn):
            self.train_indices = h5py.File(fn, 'r')['train_indices'].value
            self.val_indices = h5py.File(fn, 'r')['val_indices'].value
        else:
            print('Generating train/test split ({}/{})'.format(
                split[0], split[1]))
            n = int(np.ceil(split[0] / 100 * self.valid.sum()))
            perm = np.random.permutation(self.valid.sum())
            self.train_indices = perm[:n]
            self.val_indices = perm[n:]
            f = h5py.File(fn, 'w')
            f['train_indices'] = self.train_indices
            f['val_indices'] = self.val_indices
            f.close()

        self.train_images = self.images[self.train_indices]
        self.train_labels = self.labels[self.train_indices]
        self.val_images = self.images[self.val_indices]
        self.val_labels = self.labels[self.val_indices]
        del self.images, self.labels

        # Save actual train/val splits for visualization
        fn = src + ('/images-ch1' if self.name == 'hbp2-ch1' else '/images-ch2') + \
            '-{}-{}-train.tif'.format(split[0], split[1])
        if not os.path.isfile(fn):
            tifffile.imsave(fn, np.expand_dims(self.train_images, 1))
        fn = src + ('/images-ch1' if self.name == 'hbp2-ch1' else '/images-ch2') + \
            '-{}-{}-val.tif'.format(split[0], split[1])
        if not os.path.isfile(fn):
            tifffile.imsave(fn, np.expand_dims(self.val_images, 1))
        fn = src + ('/labels-ch1' if self.name == 'hbp2-ch1' else '/labels-ch2') + \
            '-{}-{}-train.tif'.format(split[0], split[1])
        if not os.path.isfile(fn):
            tifffile.imsave(fn, np.expand_dims(
                (self.train_labels > 1).astype(np.uint8) * 255, 1))
        fn = src + ('/labels-ch1' if self.name == 'hbp2-ch1' else '/labels-ch2') + \
            '-{}-{}-val.tif'.format(split[0], split[1])
        if not os.path.isfile(fn):
            tifffile.imsave(fn, np.expand_dims(
                            (self.val_labels > 1).astype(np.uint8) * 255, 1))
        fn = src + ('/overlay-ch1' if self.name == 'hbp2-ch1' else '/overlay-ch2') + \
            '-{}-{}-train.tif'.format(split[0], split[1])
        if not os.path.isfile(fn):
            tifffile.imsave(fn, overlay_mask(self.train_images, self.train_labels))
        fn = src + ('/overlay-ch1' if self.name == 'hbp2-ch1' else '/overlay-ch2') + \
            '-{}-{}-val.tif'.format(split[0], split[1])
        if not os.path.isfile(fn):
            tifffile.imsave(fn, overlay_mask(self.val_images, self.val_labels))

        # Relabel
        self.train_labels[self.train_labels == 255] = 1
        self.val_labels[self.val_labels == 255] = 1

        print('Training data: {}'.format(self.train_images.shape))
        print('val data: {}'.format(self.val_images.shape))

        # Store original number of slices, for valing
        self.num_train_orig = self.train_images.shape[0]

        # Offline data augmentation for biomedical data
        sim_data, sim_labels = [], []
        if self.config['augment_elastic'] == 'simard':
            src_sim = '{}/simard-{}-{}-{}'.format(
                src,
                self.config['elastic_params'][1],
                self.config['elastic_params'][2],
                self.config['elastic_params'][3])
            if not os.path.isdir(src_sim):
                os.mkdir(src_sim)

            if self.name == 'hbp2-ch1':
                root_data = src_sim + '/hbp2-ch1-data'
                root_labels = src_sim + '/hbp2-ch1-labels'
                root_vis = src_sim + '/hbp2-ch1-vis'
            elif self.name == 'hbp2-ch2':
                root_data = src_sim + '/hbp2-ch2-data'
                root_labels = src_sim + '/hbp2-ch2-labels'
                root_vis = src_sim + '/hbp2-ch2-vis'
            else:
                raise RuntimeError('Unknown dataset')

            print('Data augmentation (Simard): {} copies'.format(
                int(self.config['elastic_params'][0])))
            for i in tqdm(range(int(self.config['elastic_params'][0]))):
                fn_data = '{0:s}-{1:05d}.tif'.format(root_data, i)
                fn_labels = '{0:s}-{1:05d}.tif'.format(root_labels, i)
                fn_vis = '{0:s}-{1:05d}.tif'.format(root_vis, i)
                if os.path.isfile(fn_data) and os.path.isfile(fn_labels):
                    sim_data.append(tifffile.imread(fn_data))
                    sim_labels.append(tifffile.imread(fn_labels))
                else:
                    sim_data_i, sim_labels_i = elastic_transform(
                        self.train_images,
                        self.train_labels,
                        alpha=self.config['elastic_params'][1],
                        sigma=self.config['elastic_params'][2],
                        alpha_affine=self.config['elastic_params'][3],
                        do_3d=self.is_3d)
                    sim_data.append(sim_data_i)
                    sim_labels.append(sim_labels_i)
                    tifffile.imsave(fn_data, sim_data_i)
                    tifffile.imsave(fn_labels, sim_labels_i)
                    tifffile.imsave(fn_vis, overlay_mask(sim_data_i, sim_labels_i))

        # 2D, so make into lists
        self.train_images = [s for s in self.train_images]
        self.val_images = [s for s in self.val_images]
        self.train_labels = [s for s in self.train_labels]
        self.val_labels = [s for s in self.val_labels]
        for x in sim_data:
            for s in x:
                self.train_images.append(s)
        for x in sim_labels:
            for s in x:
                self.train_labels.append(s)
        self.test_images = None
        self.test_labels = None

        # Compute bias by slice
        self.train_mean = [s.mean() for s in self.train_images]
        self.train_std = [s.std() for s in self.train_images]
        self.val_mean = [s.mean() for s in self.val_images]
        self.val_std = [s.std() for s in self.val_images]

    def retrieve_gad_mouse_extended(self):
        src = '/cvlabdata1/cvlab/datasets_eduard/hbp-gad-mine/split-1-75-25'
        self.num_channels = 1
        self.is_3d = False
        self.dot_annotations = False
        self.evaluate_train = True
        self.evaluate_val = True
        self.evaluate_test = False
        self.plot_make_stack = True
        self.loss_label = self.name.replace('-bbp', '')

        # Load data
        if self.name == 'hbp3-ch1':
            l = 'ch1'
        elif self.name == 'hbp3-ch2':
            l = 'ch2'
        elif self.name == 'hbp3-ch2-bbp':
            l = 'ch2-bbp'
        else:
            raise RuntimeError('Error')
        self.train_images = tifffile.imread(
            src + '/images-{}-train.tif'.format(l))
        self.train_labels = tifffile.imread(
            src + '/labels-{}-train.tif'.format(l))
        self.val_images = tifffile.imread(
            src + '/images-{}-val.tif'.format(l))
        self.val_labels = tifffile.imread(
            src + '/labels-{}-val.tif'.format(l))
        self.test_images = None

        # Fill holes in labeling
        self.train_labels = self.fill_holes(self.train_labels, label='train')
        self.val_labels = self.fill_holes(self.val_labels, label='val')
        self.test_labels = None

        # Save overlay images, for visualization
        if 'bbp' not in self.name:
            fn = src + ('/vis-ch1' if self.name == 'hbp3-ch1' else '/vis-ch2') + '-train.tif'
            if not os.path.isfile(fn):
                tifffile.imsave(fn, overlay_mask(self.train_images, self.train_labels))
            fn = src + ('/vis-ch1' if self.name == 'hbp3-ch1' else '/vis-ch2') + '-val.tif'
            if not os.path.isfile(fn):
                tifffile.imsave(fn, overlay_mask(self.val_images, self.val_labels))

        # Relabel
        self.train_labels[self.train_labels == 255] = 1
        self.val_labels[self.val_labels == 255] = 1

        print('Training data: {}'.format(self.train_images.shape))
        print('Validation data: {}'.format(self.val_images.shape))

        # Store original number of slices, for testing
        self.num_train_orig = self.train_images.shape[0]

        # Data augmentation
        sim_data, sim_labels = [], []
        if self.config['augment_elastic'] == 'simard':
            src_sim = '{}/simard/{}-{}-{}'.format(
                src,
                self.config['elastic_params'][1],
                self.config['elastic_params'][2],
                self.config['elastic_params'][3])
            if not os.path.isdir(src_sim):
                os.makedirs(src_sim)

            if self.name == 'hbp3-ch1':
                root_data = src_sim + '/images-ch1'
                root_labels = src_sim + '/labels-ch1'
                root_vis = src_sim + '/vis-ch1'
            elif self.name == 'hbp3-ch2':
                root_data = src_sim + '/images-ch2'
                root_labels = src_sim + '/labels-ch2'
                root_vis = src_sim + '/vis-ch2'
            elif self.name == 'hbp3-ch2-bbp':
                root_data = src_sim + '/images-ch2-bbp'
                root_labels = src_sim + '/labels-ch2-bbp'
                root_vis = src_sim + '/vis-ch2-bpp'
            else:
                raise RuntimeError('Unknown dataset')

            print('Data augmentation (Simard): {} copies'.format(
                int(self.config['elastic_params'][0])))
            for i in tqdm(range(int(self.config['elastic_params'][0]))):
                fn_data = '{0:s}-{1:05d}.tif'.format(root_data, i)
                fn_labels = '{0:s}-{1:05d}.tif'.format(root_labels, i)
                fn_vis = '{0:s}-{1:05d}.tif'.format(root_vis, i)
                if os.path.isfile(fn_data) and os.path.isfile(fn_labels):
                    sim_data.append(tifffile.imread(fn_data))
                    sim_labels.append(tifffile.imread(fn_labels))
                else:
                    sim_data_i, sim_labels_i = elastic_transform(
                        self.train_images,
                        self.train_labels,
                        alpha=self.config['elastic_params'][1],
                        sigma=self.config['elastic_params'][2],
                        alpha_affine=self.config['elastic_params'][3],
                        do_3d=self.is_3d)
                    sim_data.append(sim_data_i)
                    sim_labels.append(sim_labels_i)
                    tifffile.imsave(fn_data, sim_data_i)
                    tifffile.imsave(fn_labels, sim_labels_i)
                    tifffile.imsave(fn_vis, overlay_mask(sim_data_i, sim_labels_i))

        # 2D, so make into lists
        self.train_images = [s for s in self.train_images]
        self.val_images = [s for s in self.val_images]
        self.train_labels = [s for s in self.train_labels]
        self.val_labels = [s for s in self.val_labels]
        for x in sim_data:
            for s in x:
                self.train_images.append(s)
        for x in sim_labels:
            for s in x:
                self.train_labels.append(s)
        self.test_images = None
        self.test_labels = None

        # Compute bias by slice
        self.train_mean = [s.mean() for s in self.train_images]
        self.train_std = [s.std() for s in self.train_images]
        self.val_mean = [s.mean() for s in self.val_images]
        self.val_std = [s.std() for s in self.val_images]

    def mirror(self, pad, images, labels):
        # Yes this is ugly
        # if self.is_3d:
        #     pad_dims = ((pad[0], pad[0]),) + ((0, 0),) + tuple((p,) * 2 for p in pad[1:])
        # else:
        #     pad_dims = ((0, 0),) + tuple((p,) * 2 for p in pad)
        pad_dims = ((0, 0),) + tuple((p,) * 2 for p in pad)

        images_padded, labels_padded = [], []
        for img, label in zip(images, labels):
            images_padded.append(np.pad(img, pad_dims, 'reflect'))
            if self.dot_annotations:
                labels_padded.append(
                    [x + pad if i < 2 else x for i, x in enumerate(labels)])
            else:
                labels_padded.append(np.pad(label, pad_dims, 'reflect'))

        return images_padded, labels_padded


class Sampler(object):
    def __init__(self, is_3d, data, config, rng, dot_annotations):
        self.is_3d = is_3d
        self.config = config
        self.data = data
        self.rng = rng
        self.dot_annotations = dot_annotations

        # Initialize sampling dicts
        self.probs = {}
        self.grid_x = {}
        self.grid_y = {}
        self.grid_z = {}

        if self.is_3d:
            self.size_z = config['input_size'][0]
            self.size_y = config['input_size'][1]
            self.size_x = config['input_size'][2]
        else:
            self.size_z = 1
            self.size_y = config['input_size'][0]
            self.size_x = config['input_size'][1]

    def sample(self):
        '''
        Sample a window from the training data.
        '''

        # Elastic deformations
        buf_z, buf_x, buf_y = 0, 0, 0
        if self.config['augment_elastic'] == 'simple':
            raise RuntimeError('Deprecated')
            # buf_z = 0
            # buf_y = int(np.round(self.rng.randn() * self.elastic_params[0]))
            # buf_x = int(np.round(self.rng.randn() * self.elastic_params[0]))

        coords = None
        if self.config['sampler'] == 'random':
            if self.is_3d:
                v = self.rng.choice(len(self.data['images']))
                z = self.rng.randint(self.data['images'][v].shape[-3] - self.size_z - buf_z)
                y = self.rng.randint(self.data['images'][v].shape[-2] - self.size_y - buf_y)
                x = self.rng.randint(self.data['images'][v].shape[-1] - self.size_x - buf_x)
                coords = (v, z, y, x)
            else:
                z = self.rng.choice(len(self.data['images']))
                y = self.rng.randint(self.data['images'][v].shape[-2] - self.size_y - buf_y)
                x = self.rng.randint(self.data['images'][v].shape[-1] - self.size_x - buf_x)
                coords = (z, y, x)

        elif self.config['sampler'] == 'uniform':
            # Account for image/stack corners being seen less often
            # First, build sampling weights if necessary
            if self.is_3d:
                raise RuntimeError('TODO')
                k = self.rng.choice(len(self.data['images']))
                n, h, w = self.data['images'][k].shape[1:]
                key = '{}x{}x{}'.format(n, h, w)
                if key not in self.probs:
                    # Build probability distribution
                    # print('Building uniform probabilities for {}x{}'.format(h, w))
                    t = time()
                    ones = np.ones((n - self.size_z + 1, h - self.size_y + 1, w - self.size_x + 1))
                    # fx = np.ones((1, self.size_x))
                    fx = np.ones((1, self.size_x))
                    fy = np.ones((self.size_y, 1))
                    weights = convolve2d(convolve2d(
                        ones, fy, mode='full'), fx, mode='full')
                    weights = 1 / weights
                    weights = weights / weights.sum()
                    # print('Done [{0:.2f} s]'.format(time() - t))

                    # print('Building uniform probabilities')
                    t = time()
                    probs = convolve2d(
                        convolve2d(
                            weights,
                            np.ones((1, self.size_x)), mode='valid'),
                        np.ones((self.size_y, 1)), mode='valid')
                    probs = probs / probs.sum()
                    self.probs[key] = probs

                    # Mapping from indices to 2D coordinates
                    grid = np.meshgrid(
                        np.arange(self.probs[key].shape[1]),
                        np.arange(self.probs[key].shape[0])
                    )
                    self.grid_x[key] = grid[0].flatten()
                    self.grid_y[key] = grid[1].flatten()
                    print('Built uniform probabilities for {0:d}x{1:d} [{2:.2f} s]'.format(h, w, time() - t))
                    # print('Done [{0:.2f} s]'.format(time() - t))

                # Actually sample
                raise RuntimeError('TODO')
            else:
                z = self.rng.choice(len(self.data['images']))
                h, w = self.data['images'][z].shape[-2], self.data['images'][z].shape[-1]
                key = '{}x{}'.format(h, w)
                if key not in self.probs:
                    # Build probability distribution
                    # print('Building uniform probabilities for {}x{}'.format(h, w))
                    t = time()
                    # self.weights = np.zeros((h, w))
                    # for y in range(0, self.data['images'].shape[2] - self.size_y + 1):
                    #     for x in range(0, self.data['images'].shape[3] - self.size_x + 1):
                    #         self.weights[y:y + self.size_y, x:x + self.size_x] += 1
                    ones = np.ones((h - self.size_y + 1, w - self.size_x + 1))
                    fx = np.ones((1, self.size_x))
                    fy = np.ones((self.size_y, 1))
                    weights = convolve2d(convolve2d(
                        ones, fy, mode='full'), fx, mode='full')
                    weights = 1 / weights
                    weights = weights / weights.sum()
                    # print('Done [{0:.2f} s]'.format(time() - t))

                    # print('Building uniform probabilities')
                    t = time()
                    probs = convolve2d(
                        convolve2d(
                            weights,
                            np.ones((1, self.size_x)), mode='valid'),
                        np.ones((self.size_y, 1)), mode='valid')
                    probs = probs / probs.sum()
                    self.probs[key] = probs

                    # Mapping from indices to 2D coordinates
                    grid = np.meshgrid(
                        np.arange(self.probs[key].shape[1]),
                        np.arange(self.probs[key].shape[0])
                    )
                    self.grid_x[key] = grid[0].flatten()
                    self.grid_y[key] = grid[1].flatten()
                    print('Built uniform probabilities for {0:d}x{1:d} [{2:.2f} s]'.format(h, w, time() - t))
                    # print('Done [{0:.2f} s]'.format(time() - t))

                # Actually sample
                k = np.random.choice(
                    np.arange(self.grid_x[key].size), p=self.probs[key].flatten())
                coords = (z, self.grid_y[key][k], self.grid_x[
                          key][k], self.probs[key].flatten()[k])

        elif self.config['sampler'] == 'fg_only_flat' or self.config['sampler'] == 'fg_only_prob':
            # Sample patches with foreground cells only
            # Use constant weights for now (regardless of the number of voxels in the patch)
            if self.is_3d:
                # Build weights (for each image/stack)
                v = self.rng.choice(len(self.data['images']))
                if v not in self.probs:
                    # # Scan labels
                    # l = self.data['labels'][v].copy()
                    # l[l == 2] = 0
                    # 
                    # # Build probability distribution
                    # t = time()
                    # f = torch.nn.Conv3d(1, 1, (self.size_z,
                    #                            self.size_y, self.size_x))
                    # f.weight.data[:] = 0
                    # f.bias.data[:] = 0
                    # # Margins are lost, ignore them
                    # m = [p // 2 for p in self.config['margin']]
                    # f.weight.data[
                    #     :,
                    #     :,
                    #     m[0]:self.size_z - m[0],
                    #     m[1]:self.size_y - m[1],
                    #     m[2]:self.size_x - m[2]] = 1
                    # weights = f.cuda().forward(Variable(torch.from_numpy(
                    #     l[None, ...]).float().cuda())).data[0][0].cpu()

                    # 3D convolutions are too slow, use a summed-area table
                    # Data (turn off ignored pixels)
                    l = self.data['labels'][v][0].copy()
                    l[l == 2] = 0

                    fs_out = self.config['output_size'][0]
                    fs_in = self.config['input_size'][0]
                    s = l.shape

                    print('Extracting probability distribution from labels: '
                          '{}x{}x{}'.format(s[0] - fs_in + 1,
                                            s[1] - fs_in + 1,
                                            s[2] - fs_in + 1))
                    t = time()

                    # Filter size
                    if len(np.unique(self.config['output_size'])) > 1:
                        raise RuntimeError('This works only for NxNxN (fix otherwise)')

                    # Compute the summed-area table
                    sat = l.astype(np.int64).cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)

                    # Add a row/column of zeros for conveniency
                    # so we can call instum with a=(0, 0, 0)
                    sat0 = np.zeros([_s + 1 for _s in sat.shape])
                    sat0[1:, 1:, 1:] = sat

                    # Returns the area determined by (a[0], a[1], a[2])
                    # and (b[0], b[1], b[2])
                    def intsum(t, a, b):
                        b = [_b + 1 for _b in b]
                        return t[b[0], b[1], b[2]] + \
                            t[a[0], a[1], b[2]] + \
                            t[a[0], b[1], a[2]] + \
                            t[b[0], a[1], a[2]] - \
                            t[b[0], b[1], a[2]] - \
                            t[b[0], a[1], b[2]] - \
                            t[a[0], b[1], b[2]] - \
                            t[a[0], a[1], a[2]]

                    # Skip the margins
                    # m = [0, 0, 0]
                    m = [m // 2 for m in self.config['margin']]

                    # Equivalent to the convolution
                    weights = np.zeros([s[0] - fs_in + 1,
                                        s[1] - fs_in + 1,
                                        s[2] - fs_in + 1],
                                       dtype=np.int64)
                    for z in range(weights.shape[0]):
                        for y in range(weights.shape[1]):
                            for x in range(weights.shape[2]):
                                weights[z, y, x] = intsum(
                                    sat0,
                                    (z + m[0], y + m[1], x + m[2]),
                                    (z + m[0] + fs_out - 1, y + m[1] + fs_out - 1, x + m[2] + fs_out - 1))

                    # We don't need the margins
                    # This could be optimized but I'm lazy
                    # weights = weights[m[0]:-m[0], m[1]:-m[1], m[2]:-m[2]]

                    # Make flat if that's what we want
                    if self.config['sampler'] == 'fg_only_flat':
                        weights[weights > 0] = 1
                    self.probs[v] = weights / weights.sum()

                    # Mapping from indices to 3D coordinates
                    grid = np.meshgrid(
                        np.arange(self.probs[v].shape[0]),
                        np.arange(self.probs[v].shape[1]),
                        np.arange(self.probs[v].shape[2]),
                    )
                    self.grid_z[v] = grid[0].flatten()
                    self.grid_y[v] = grid[1].flatten()
                    self.grid_x[v] = grid[2].flatten()
                    print('Built {:s} probabilities for stack {:d} (of {:d}) [{:.2f} s]'.format(
                        'fg/flat' if self.config['sampler'] == 'fg_only_flat' else 'fg/prob',
                        v + 1,
                        len(self.data['images']),
                        time() - t))

                    # Flatten probabilities
                    self.probs[v] = self.probs[v].flatten()

                # Actually sample
                k = np.random.choice(np.arange(self.grid_x[v].size),
                                     p=self.probs[v])
                coords = (v,
                          self.grid_z[v][k],
                          self.grid_y[v][k],
                          self.grid_x[v][k],
                          self.probs[v][k])
            else:
                # Build weights (for each image/stack)
                z = self.rng.choice(len(self.data['images']))
                if z not in self.probs:
                    # Scan labels
                    l = self.data['labels'][z].copy()
                    l[l == 2] = 0

                    # Build probability distribution
                    t = time()
                    fx = np.ones((1, self.size_x))
                    fy = np.ones((self.size_y, 1))
                    # Margins are lost, ignore them
                    m = [m // 2 for m in self.config['margin']]
                    fx[:, :m[0]] = 0
                    fx[:, self.size_y - m[1]:] = 0
                    fy[:m[1], :] = 0
                    fy[self.size_y - m[1]:, :] = 0
                    weights = convolve2d(convolve2d(
                        l[0], fy, mode='valid'), fx, mode='valid')

                    # Make flat if that's what we want
                    if self.config['sampler'] == 'fg_only_flat':
                        weights[weights > 0] = 1
                    self.probs[z] = weights / weights.sum()

                    # Mapping from indices to 2D coordinates
                    grid = np.meshgrid(
                        np.arange(self.probs[z].shape[1]),
                        np.arange(self.probs[z].shape[0])
                    )
                    self.grid_x[z] = grid[0].flatten()
                    self.grid_y[z] = grid[1].flatten()
                    print('Built {:s} probabilities for slice {:d} (of {:d}) [{:.2f} s]'.format(
                        'fg/flat' if self.config['sampler'] == 'fg_only_flat' else 'fg/prob',
                        z + 1,
                        len(self.data['images']),
                        time() - t))

                # Actually sample
                k = np.random.choice(
                    np.arange(self.grid_x[z].size), p=self.probs[z].flatten())
                coords = (z, self.grid_y[z][k], self.grid_x[
                          z][k], self.probs[z].flatten()[k])
        else:
            raise RuntimeError('Unknown sampler type')

        # Sample
        if self.is_3d:
            # Images
            b = self.data['images'][coords[0]][
                :,
                coords[1]:coords[1] + self.size_z + buf_z,
                coords[2]:coords[2] + self.size_y + buf_y,
                coords[3]:coords[3] + self.size_x + buf_x,
            ][None, ...].copy().astype(np.float)

            # Labels
            if self.dot_annotations:
                raise RuntimeError('TODO')
            else:
                l = self.data['labels'][coords[0]][
                    :,
                    coords[1]:coords[1] + self.size_z + buf_z,
                    coords[2]:coords[2] + self.size_y + buf_y,
                    coords[3]:coords[3] + self.size_x + buf_x,
                ][None, ...].copy()
        else:
            # Images
            b = self.data['images'][coords[0]][
                :,
                coords[1]:coords[1] + self.size_y + buf_y,
                coords[2]:coords[2] + self.size_x + buf_x,
            ][None, ...].copy().astype(np.float)

            # Labels
            if self.dot_annotations:
                # Points outside the image will be ignored by the loss function
                l = self.data['labels'][coords[0]] - coords[2:0:-1]

                # Will require scaling if we allow zooming in for data augmentation
                # ...

                # This is not necessary, but we might want to eliminate OOB points later on
                # l = l[np.bitwise_and(
                #     np.bitwise_and(l[:,0] > 0, l[:,1] > 0),
                #     np.bitwise_and(l[:,0] < self.output_size[0], l[:,1] < self.output_size[1]))]
            else:
                l = self.data['labels'][coords[0]][
                    :,
                    coords[1]:coords[1] + self.size_y + buf_y,
                    coords[2]:coords[2] + self.size_x + buf_x,
                ][None, ...].copy()

        # embed()
        # Rotate sample
        if self.config['augment_rotation'] == '2d':
            b, l = self.rotate_sample_2d(b, l)
        elif self.config['augment_rotation'] != 'none':
            raise RuntimeError('Flipping: unknown type')

        # Flip
        if self.config['augment_flipping'] == 'zyx':
            b, l = self.flip_sample(b, l, [True] * 3)
        elif self.config['augment_flipping'] == 'yx':
            b, l = self.flip_sample(b, l, [False, True, True])
        elif self.config['augment_flipping'] == 'y':
            b, l = self.flip_sample(b, l, [False, True, False])
        elif self.config['augment_flipping'] == 'x':
            b, l = self.flip_sample(b, l, [False, False, True])
        elif self.config['augment_flipping'] != 'none':
            raise RuntimeError('Flipping: unknown type')

        # Elastic deformations
        # if self.elastic == 'simple':
        #     b = imresize(b[0][0],
        #                  (self.size_y, self.size_x),
        #                  interp='bicubic',
        #                  mode='F').astype(np.float)[None, None, ...]
        #     # Labels can be empty (i.e. flat), in which case scipy's resize fails
        #     if l.max() - l.min() > 0:
        #         # mode 'I' does not work as expected, use 'F' and cast
        #         l = imresize(l[0],
        #                      (self.size_y, self.size_x),
        #                      interp='nearest',
        #                      mode='F').astype(l.dtype)[None, ...]
        #     else:
        #         l = np.zeros((self.size_z, self.size_y, self.size_x),
        #                      dtype=l.dtype) + l.flatten()[0]

        # Normalize
        if self.config['use_lcn']:
            b = (b - b.mean()) / b.std()
        else:
            b = (b - self.data['mean'][coords[0]]) / self.data['std'][coords[0]]

        # Remove padding from the labels
        # Must happen after 'simple' elastic deformations
        if not self.dot_annotations:
            if self.is_3d:
                l = l[:,
                      :,
                      self.config['margin'][-3] // 2:-(self.config['margin'][-3] // 2),
                      self.config['margin'][-2] // 2:-(self.config['margin'][-2] // 2),
                      self.config['margin'][-1] // 2:-(self.config['margin'][-1] // 2)]
            else:
                l = l[:,
                      :,
                      self.config['margin'][-2] // 2:-(self.config['margin'][-2] // 2),
                      self.config['margin'][-1] // 2:-(self.config['margin'][-1] // 2)]
        # else:
        #     l = l - np.array(self.config['margin'][1:]
        #                      )[None, ...].repeat(l.shape[0], axis=0)

        if self.config['augment_elastic'] == 'none':
            return b, l, coords, None
        elif self.config['augment_elastic'] == 'simple':
            return b, l, coords, (buf_z, buf_y, buf_x)
        elif self.config['augment_elastic'] == 'simard':
            return b, l, coords, 0

    def rotate_sample_2d(self, data, labels):
        '''
        Rotate patch or stack by 90 degree increments.
        Will process each slice separately if given a 3D sample.
        '''
        num_rots = self.rng.randint(4)
        if num_rots > 0:
            data = np.rot90(data, num_rots, axes=(data.ndim - 2, data.ndim - 1))
            if not self.dot_annotations:
                labels = np.rot90(labels, num_rots, axes=(labels.ndim - 2, labels.ndim - 1))
            else:
                # rot = np.array([
                #     [np.cos(np.pi * num_rots / np.pi), -np.sin(np.pi * num_rots / np.pi)],
                #     [np.sin(np.pi * num_rots / np.pi), np.cos(np.pi * num_rots / np.pi)]])
                raise RuntimeError('Not doing this now')
            return data.copy(), labels.copy()
        else:
            return data, labels

    def flip_sample(self, data, labels, dims):
        '''
        Flip data/labels over each axis.
        '''
        toss_z, toss_y, toss_x = [False] * 3
        if dims[2]:
            toss_x = self.rng.rand() >= .5
            if toss_x:
                data = data[:, :, :, ::-1]
                if not self.dot_annotations:
                    labels = labels[:, :, :, ::-1]
                else:
                    labels[:, 0] = data.shape[-2] - labels[:, 0]

        if dims[1]:
            toss_y = self.rng.rand() >= .5
            if toss_y:
                data = data[:, :, ::-1, :]
                if not self.dot_annotations:
                    labels = labels[:, :, ::-1, :]
                else:
                    labels[:, 1] = data.shape[-1] - labels[:, 1]

        if dims[0]:
            toss_z = self.rng.rand() >= .5
            if toss_z:
                data = data[:, ::-1, :, :]
                if not self.dot_annotations:
                    labels = labels[:, ::-1, :, :]
                else:
                    labels[:, 1] = data.shape[-1] - labels[:, 1]

        # print('Toss is {}, {}'.format(toss_h, toss_v))
        # Pytorch currently does not support numpy -> tensor with negative strides
        if any([toss_z, toss_y, toss_x]):
            return data.copy(), labels.copy()
        else:
            return data, labels

# class PatchImportanceSampler(object):
#     
#     class CenterImportanceSampler(object):
#         
#         def __init__(self, _importance, in_patch_shape, rng):
#             
#             # Mask the importance given the input patch size
#             margin1 = tuple(ps // 2 for ps in in_patch_shape)
#             margin2 = tuple(ps - ps // 2 - 1 for ps in in_patch_shape)
#             slices = tuple(slice(m1, -m2 if m2 != 0 else None) for m1, m2 in zip(margin1, margin2))
#             importance = np.zeros_like(_importance)
#             importance[slices] = _importance[slices]
#             
#             self.img_size = importance.size
#             self.img_shape = importance.shape
#             
#             self.indices = np.arange(self.img_size)
#             importance = importance.flatten()
#             importance = np.cumsum(importance)
#             importance /= importance[-1]
#             
#             self.importance = importance
#             self.rng = rng
#         
#         def sample_center(self):
#             aux = self.rng.random_sample()
#             index = self.indices[np.searchsorted(self.importance, aux, side='right')]
#             center = np.unravel_index(index, dims=self.img_shape)
#             
#             return center
#     
#     def __init__(self, unet_config,
#                  training_x, training_y,
#                  patch_shape,
#                  loss_weights,
#                  sampling_weights,
#                  transformations=[lambda x: x],
#                  mask_func=np.isnan,
#                  rng=None):
# 
#         for ty_i in training_y:
#             if any([a > b for a, b in zip(patch_shape, ty_i.shape)]):
#                 raise ValueError("The patch_shape {} is larger than the shape {} of one of the training images.".format(patch_shape, ty_i.shape))
# 
#         if not (len(training_x) == len(training_y) == len(sampling_weights) == len(loss_weights)):
#             raise ValueError("The length of `training_x`, `training_y`, `sampling_weights` and `loss_weights` must be equal.")
#         
#         if not all([i.shape[:unet_config.ndims] == j.shape == k.shape == m.shape for i, j, k, m in zip(training_x, training_y, sampling_weights, loss_weights)]):
#             raise ValueError("The shape of `training_x`, `training_y`, `sampling_weights` and `loss_weights` must be equal.")
#         
#         if rng is None:
#             rng = np.random.RandomState()
#         self.rng = rng
#         
#         num_classes = unet_config.num_classes
#         if num_classes > 1:
#             self.training_counter = BinCounter(num_classes + 1)
#             self.weighted_counter = BinCounter(num_classes + 1)
#         
#         self.unet_config = unet_config
#         self.training_x = training_x
#         self.training_y = training_y
#         
#         self.margin = self.unet_config.margin()
#         self.patch_shape = patch_shape
#         self.in_patch_shape = self.patch_shape + 2 * self.margin
#         self.transformations = transformations
#         self.mask_func = mask_func
# 
#         self.iters_per_epoch = len(self.training_x) * len(self.transformations)
#         self.reset()
# 
#         self.loss_weights = loss_weights
#         self.update_sampling_weights( sampling_weights )
# 
# 
#     def update_sampling_weights(self, weights):
#         """ Updates sampling weights. """
#         """ Make sure to pad them before calling this function """
#         self.sampling_weights = weights
# 
# 
#         patch_importances = [ ndimage.uniform_filter(weights[i], self.patch_shape, mode='constant')
#                                 for i in range(len(self.sampling_weights))]
# 
#         self.samplers = [self.CenterImportanceSampler(i, self.in_patch_shape, self.rng)
#                             for i in patch_importances]
# 
# 
#         # origin needs to be corrected if patch size is even
#         fixed_origin = [ -1 if p % 2 == 0 else 0 for p in self.patch_shape ]
#         pix_sampling_prob = [ndimage.uniform_filter(i, self.patch_shape,
#                                mode='constant', origin=fixed_origin)
#                                 for i in patch_importances]
# 
#         self.pix_sampling_prob = pix_sampling_prob
# 
#         # compute normalization weights
#         norm_weights = [ pix_sampling_prob[i].size / np.sum(pix_sampling_prob[i])
#                             for i in range(len(patch_importances))]
# 
#         self.sampling_correction_factor = [self.loss_weights[i] / (1e-9 + norm_weights[i] * pix_sampling_prob[i])
#                                            for i in range(len(patch_importances))]
# 
#     def get_minibatch(self, index):
# 
#         patch_x, patch_y, patch_w = self.sample_patch(index)
#         patch_y = np.copy(patch_y)
#         patch_w = patch_w.astype(np.float32)
# 
#         mask = self.mask_func(patch_y)
#         patch_y[mask] = 0
#         patch_w[mask] = 0
# 
#         if not np.issubdtype(patch_y.dtype, float):
#             # classification?
#             self.training_counter.update(patch_y)
#             self.weighted_counter.update(patch_y, patch_w)
# 
#         return patch_x, patch_y, patch_w
# 
#     def sample_patch(self, index):
#         
#         index = self.indices[index % self.iters_per_epoch]
#         transform_idx, sample_idx  = divmod(index, len(self.training_x))
#         
#         current_x = self.training_x[sample_idx]
#         current_y = self.training_y[sample_idx]
#         current_w = self.sampling_correction_factor[sample_idx]
#         
#         center = self.samplers[sample_idx].sample_center()
#         patch_y = patch_utils.get_patch(current_y, self.patch_shape, center)
#         patch_x = patch_utils.get_patch(current_x, self.in_patch_shape, center)
#         patch_w = patch_utils.get_patch(current_w, self.patch_shape, center)
#         
#         transform = self.transformations[transform_idx]
#         patch_x = transform(patch_x)
#         patch_y = transform(patch_y)
#         patch_w = transform(patch_w)
#         
#         return patch_x, patch_y, patch_w
#     
#     def reset(self):
#         self.indices = np.arange(self.iters_per_epoch)
#     
#     def shuffle(self):
#         self.rng.shuffle(self.indices)
#     
#     def save_state(self, filename):
#         state = dict(indices=self.indices)
#         
#         if self.unet_config.num_classes > 1:
#             state["training_counter_counts"] = self.training_counter.counts
#             state["weighted_counter_counts"] = self.weighted_counter.counts
#         
#         np.savez(filename, **state)
#     
#     def load_state(self, filename):
#         state = np.load(filename)
#         
#         self.indices = state["indices"]
#         
#         if self.unet_config.num_classes > 1:
#             self.training_counter.counts = state["training_counter_counts"]
#             self.weighted_counter.counts = state["weighted_counter_counts"]
# 
# 
# def invfreq_importance(labels, patch_shape, label_frequencies=None):
#     """
#     The importance of every patch is inversely proportional to the frequency
#     of the classes it contains.
#     """
# 
#     if label_frequencies is None:
#         counter = BinCounter(x=labels)
#         label_frequencies = counter.frequencies
# 
#     num_labels = len(label_frequencies)
# 
#     counts = []
#     for i in xrange(num_labels):
#         counts_i = np.float_(labels == i)
#         counts_i = ndimage.uniform_filter(counts_i, patch_shape, mode='constant')
#         counts.append(counts_i)
# 
#     counts = np.asarray(counts)
#     freqs = counts / np.sum(counts, 0)
# 
#     label_frequencies = np.copy(label_frequencies)
#     label_frequencies[label_frequencies == 0] = 1e-8
#     slices = (slice(None),) + (None,) * labels.ndim
#     importance = np.sum(freqs / label_frequencies[slices], 0)
# 
#     return importance

import deepdish as dd
import tifffile
from tqdm import tqdm
import numpy as np
from IPython import embed
import os
from skimage import measure


# do_what = 'standard'
# do_what = 'masked_ch2'
do_what = 'ch1_3d'

if do_what == 'standard':
    src = '/cvlabdata1/cvlab/datasets_eduard/hbp-gad-mine'
    split_id = 1
    splits = [75, 25]

    # read data/labels
    images = []
    images.append(tifffile.imread(src + '/raw/images-ch1.tif'))
    images.append(tifffile.imread(src + '/raw/images-ch2.tif'))
    labels = []
    labels.append(tifffile.imread(src + '/raw/labels-ch1-full.tif'))
    labels.append(tifffile.imread(src + '/raw/labels-ch2-full.tif'))

    # avoid rgb confusion when z-dim is 3
    for j in [0, 1]:
        images[j] = np.expand_dims(images[j], 1)
        labels[j] = np.expand_dims(labels[j], 1)

    # crop
    images[0] = images[0][list(range(0, 31, 2)) + list(range(31, 48, 2)) + [50, 52], :, :990, :1708]
    labels[0] = labels[0][list(range(0, 31, 2)) + list(range(31, 48, 2)) + [50, 52], :, :990, :1708]
    images[1] = images[1][list(range(0, 21, 2)) + [21], :, :1170, :1590]
    labels[1] = labels[1][list(range(0, 21, 2)) + [21], :, :1170, :1590]

    # stats
    d = {'Channel 1': labels[0], 'Channel 2': labels[1]}
    num = {}
    for k in d:
        num[k] = []
        for i, j in enumerate(d[k]):
            num[k].append(measure.label(j, background=0).max())
            print('{}, slice {}: {} fg pixels, {} connected components'.format(k, i, j.sum(), num[k][-1]))
    for k in d:
        print('{0:s}: {1:d} CC total, avg/slice {2:.3f}'.format(k, sum(num[k]), sum(num[k]) / len(num[k])))

    # re-format the labels, for visualization
    labels[0][labels[0] == 1] = 255
    labels[1][labels[1] == 1] = 255

    # save full images
    if not os.path.isdir(src + '/cropped'):
        os.makedirs(src + '/cropped')
    tifffile.imsave(src + '/cropped/images-ch1.tif', images[0])
    tifffile.imsave(src + '/cropped/images-ch2.tif', images[1])
    tifffile.imsave(src + '/cropped/labels-ch1.tif', labels[0])
    tifffile.imsave(src + '/cropped/labels-ch2.tif', labels[1])

    # make split
    split_dir = src + '/split-{}-'.format(split_id) + '-'.join([str(s) for s in splits])
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
    perm1 = np.random.permutation(images[0].shape[0])
    perm2 = np.random.permutation(images[1].shape[0])

    num_val_ch1 = int(np.ceil(splits[1] / 100 * perm1.size))
    num_val_ch2 = int(np.ceil(splits[1] / 100 * perm2.size))

    tifffile.imsave(split_dir + '/images-ch1-val.tif', images[0][perm1[:num_val_ch1]])
    tifffile.imsave(split_dir + '/images-ch2-val.tif', images[1][perm2[:num_val_ch2]])
    tifffile.imsave(split_dir + '/labels-ch1-val.tif', labels[0][perm1[:num_val_ch1]])
    tifffile.imsave(split_dir + '/labels-ch2-val.tif', labels[1][perm2[:num_val_ch2]])
    tifffile.imsave(split_dir + '/images-ch1-train.tif', images[0][perm1[num_val_ch1:]])
    tifffile.imsave(split_dir + '/images-ch2-train.tif', images[1][perm2[num_val_ch2:]])
    tifffile.imsave(split_dir + '/labels-ch1-train.tif', labels[0][perm1[num_val_ch1:]])
    tifffile.imsave(split_dir + '/labels-ch2-train.tif', labels[1][perm2[num_val_ch2:]])

    # dd.io.save('{}/data-split-minsize-{}-tr-{}-val-{}.h5'.format(src, minsize, splits[0], splits[1]), data)
    # print('Done!')

elif do_what == 'masked_ch2':
    src_images = '/cvlabdata1/cvlab/datasets_eduard/gad-mouse/stacks2-masked'
    src_labels = '/cvlabdata1/cvlab/datasets_eduard/hbp-gad-mine'
    split_id = 1
    splits = [75, 25]

    # read data/labels
    images = tifffile.imread(src_images + '/stack_ch2_range.tiff')
    labels = tifffile.imread(src_labels + '/raw/labels-ch2-full.tif')

    # avoid rgb confusion when z-dim is 3
    images = np.expand_dims(images, 1)
    labels = np.expand_dims(labels, 1)

    # crop
    images = images[list(range(0, 21, 2)) + [21], :, :1170, :1590]
    labels = labels[list(range(0, 21, 2)) + [21], :, :1170, :1590]

    # re-format the labels, for visualization
    labels[labels == 1] = 255

    # save full images
    if not os.path.isdir(src_labels + '/cropped'):
        os.makedirs(src_labels + '/cropped')
    tifffile.imsave(src_labels + '/cropped/images-ch2-bbp.tif', images)
    # tifffile.imsave(src_labels + '/cropped/labels-ch2-bbp.tif', labels)

    # make split
    split_dir = src_labels + '/split-{}-'.format(split_id) + '-'.join([str(s) for s in splits])
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
    perm = np.array([6, 3, 10, 5, 12, 11, 9, 7, 1, 2, 4, 8]) - 1

    num_val_ch2 = int(np.ceil(splits[1] / 100 * perm.size))

    tifffile.imsave(split_dir + '/images-ch2-bbp-val.tif', images[perm[:num_val_ch2]])
    tifffile.imsave(split_dir + '/labels-ch2-bbp-val.tif', labels[perm[:num_val_ch2]])
    tifffile.imsave(split_dir + '/images-ch2-bbp-train.tif', images[perm[num_val_ch2:]])
    tifffile.imsave(split_dir + '/labels-ch2-bbp-train.tif', labels[perm[num_val_ch2:]])

elif do_what == 'ch1_3d':
    # src_images = '/cvlabdata1/cvlab/datasets_eduard/gad-mouse/stacks2'
    # src_labels = '/cvlabdata1/cvlab/datasets_eduard/hbp-gad-mine'
    # split_id = 1
    # splits = [75, 25]
    # 
    # # read data/labels
    # images = tifffile.imread(src_images + '/stack_ch2.tiff')
    # labels = tifffile.imread(src_labels + '/raw/labels-ch2-full.tif')
    # 
    # # avoid rgb confusion when z-dim is 3
    # images = np.expand_dims(images, 1)
    # labels = np.expand_dims(labels, 1)
    # 
    # # crop
    # images = images[list(range(0, 21, 2)) + [21], :, :1170, :1590]
    # labels = labels[list(range(0, 21, 2)) + [21], :, :1170, :1590]
    # 
    # # re-format the labels, for visualization
    # labels[labels == 1] = 255
    # 
    # # save full images
    # if not os.path.isdir(src_labels + '/cropped'):
    #     os.makedirs(src_labels + '/cropped')
    # tifffile.imsave(src_labels + '/cropped/images-ch2-bbp.tif', images)
    # # tifffile.imsave(src_labels + '/cropped/labels-ch2-bbp.tif', labels)
    # 
    # # make split
    # split_dir = src_labels + '/split-{}-'.format(split_id) + '-'.join([str(s) for s in splits])
    # if not os.path.isdir(split_dir):
    #     os.makedirs(split_dir)
    # perm = np.array([6, 3, 10, 5, 12, 11, 9, 7, 1, 2, 4, 8]) - 1
    # 
    # num_val_ch2 = int(np.ceil(splits[1] / 100 * perm.size))
    # 
    # tifffile.imsave(split_dir + '/images-ch2-bbp-val.tif', images[perm[:num_val_ch2]])
    # tifffile.imsave(split_dir + '/labels-ch2-bbp-val.tif', labels[perm[:num_val_ch2]])
    # tifffile.imsave(split_dir + '/images-ch2-bbp-train.tif', images[perm[num_val_ch2:]])
    # tifffile.imsave(split_dir + '/labels-ch2-bbp-train.tif', labels[perm[num_val_ch2:]])

else:
    raise RuntimeError("S-sorry?")

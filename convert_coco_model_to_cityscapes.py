#!/usr/bin/env python

# Copyright (c) 2020-present, Groupe Renault, SAS.
##############################################################################

# Convert a detection model trained for COCO into a model that can be fine-tuned
# on cityscapes
#
# cityscapes_to_coco

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import sys
import h5py 

import detectron.datasets.coco_to_cityscapes_id as cs

NUM_CS_CLS = 9
NUM_COCO_CLS = 81


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a COCO pre-trained model for use with Cityscapes')
    parser.add_argument(
        '--coco_model', dest='coco_model_file_name',
        help='Pretrained network weights file path',
        default=None, type=str)
    parser.add_argument(
        '--convert_func', dest='convert_func',
        help='Blob conversion function',
        default='cityscapes_to_coco', type=str)
    parser.add_argument(
        '--output', dest='out_file_name',
        help='Output file path',
        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def load_object(file_name):
    return h5py.File(file_name, 'r')

def create_file(file_name):
    return h5py.File(file_name, 'a')

def traverse_datasets(hdf_file, new_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                new_file.require_dataset(path, shape=item.shape, dtype=item.dtype, data=item, maxshape=item.shape, chunks=True)
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                new_file.require_group(path)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path

def convert_coco_layers_to_cityscape_layers(model_dict, new_model):
    for dset in traverse_datasets(model_dict, new_model):
        v = model_dict[dset]
        print('Path:', dset)
        print('Shape:', v.shape)
        print('Data type:', v.dtype)
        if v.shape[0] == NUM_COCO_CLS or v.shape[0] == 4 * NUM_COCO_CLS:
            coco_layer = model_dict[dset][()]
            print(
                'Converting COCO layer {} with shape {}'.
                format(dset, coco_layer.shape)
            )
            cs_layer = convert_coco_layer_to_cityscapes_layer(
                coco_layer, args.convert_func
            )
            print(' -> converted shape {}'.format(cs_layer.shape))
            new_model[dset].resize(cs_layer.shape)
            new_model[dset][:] = cs_layer


def convert_coco_layer_to_cityscapes_layer(coco_layer, convert_func):
    # coco blob (81, ...) or (81*4, ...)
    coco_shape = coco_layer.shape
    leading_factor = int(coco_shape[0] / NUM_COCO_CLS)
    tail_shape = list(coco_shape[1:])
    assert leading_factor == 1 or leading_factor == 4

    # Reshape in [num_classes, ...] form for easier manipulations
    coco_layer = coco_layer.reshape([NUM_COCO_CLS, -1] + tail_shape)
    # Default initialization uses Gaussian with mean and std to match the
    # existing parameters
    std = coco_layer.std()
    mean = coco_layer.mean()
    cs_shape = [NUM_CS_CLS] + list(coco_layer.shape[1:])
    cs_layer = (np.random.randn(*cs_shape) * std + mean).astype(np.float32)

    # Replace random parameters with COCO parameters if class mapping exists
    for i in range(NUM_CS_CLS):
        coco_cls_id = getattr(cs, convert_func)(i)
        if coco_cls_id >= 0:  # otherwise ignore (rand init)
            cs_layer[i] = coco_layer[coco_cls_id]

    cs_shape = [NUM_CS_CLS * leading_factor] + tail_shape
    return cs_layer.reshape(cs_shape)

def load_and_convert_coco_model(args):
    model_dict = load_object(args.coco_model_file_name)
    new_model = create_file(args.out_file_name)
    convert_coco_layers_to_cityscape_layers(model_dict, new_model)
    return new_model


if __name__ == '__main__':
    args = parse_args()
    print(args)
    assert os.path.exists(args.coco_model_file_name), \
        'Weights file does not exist'
    weights = load_and_convert_coco_model(args)
    print('Wrote layers to {}:'.format(args.out_file_name))
    print(sorted(weights.keys()))

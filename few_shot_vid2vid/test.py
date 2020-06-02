# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os
import numpy as np
import torch
import cv2
from collections import OrderedDict

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from util.util import save_image

def initialize(opt=None):
    if opt is None:
        opt = TestOptions().parse()

    # setup dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # setup models
    model = create_model(opt)
    model.eval()
    visualizer = Visualizer(opt)

    return opt, model, visualizer, dataset

def evaluate_many():
    opt = TestOptions().parse()
    opt.name = "street"
    opt.dataset_mode = "fewshot_street"
    opt.checkpoints_dir = "./few_shot_vid2vid/checkpoints/"
    opt.adaptive_spade = True
    opt.loadSize = 512
    opt.fineSize = 512
    opt, model, visualizer, dataset = initialize()
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name,
                        '%s_%s' % (opt.phase, opt.which_epoch))
    if opt.finetune:
        web_dir += '_finetune'
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
        opt.name, opt.phase, opt.which_epoch), infer=True)

    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many or i >= len(dataset):
            break
        img_path = data['path']
        data_list = [data['tgt_label'], data['tgt_image'], None,
                    None, data['ref_label'], data['ref_image'], None, None]
        synthesized_image, _, _, _, _, _ = model(data_list)

        synthesized_image = util.tensor2im(synthesized_image)
        tgt_image = util.tensor2im(data['tgt_image'])
        ref_image = util.tensor2im(data['ref_image'], tile=True)
        seq = data['seq'][0]
        visual_list = [ref_image, tgt_image, synthesized_image]
        visuals = OrderedDict([(seq, np.hstack(visual_list)),
                            (seq + '/synthesized', synthesized_image),
                            (seq + '/ref_image', ref_image if i == 0 else None),
                            ])
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

def infer_images(seg_img_dataroot, target_img_path, save_path):
    '''Launch the inference of label images using the pre-trained neural
    network.'''
    opt = TestOptions().parse()
    target_img_name = target_img_path.split("/")[-1]
    opt.name = "street"
    opt.dataset_mode = "fewshot_street_prod"
    opt.checkpoints_dir = "./checkpoints/"
    opt.dataroot = seg_img_dataroot
    opt.adaptive_spade = True
    opt.loadSize = 512
    opt.fineSize = 512
    opt.seq_path = seg_img_dataroot
    opt.ref_img_path = target_img_path.replace(target_img_name, "")
    opt.label_nc = 20
    opt.input_nc = 3
    opt.aspect_ratio = 2
    opt.resize_or_crop = 'scale_width_and_crop'
    opt, model, _, dataset = initialize(opt)

    os.makedirs(save_path, exist_ok=True)

    for i, data in enumerate(dataset):
            data_list = [data['tgt_label'], None, None,
                    None, data['ref_label'], data['ref_image'], None, None, None]
            synthesized_image, _, _, _, _, _ = model(data_list)
            synthesized_image = util.tensor2im(synthesized_image)
            save_image(synthesized_image, os.path.join(save_path, str(i) + '.png'))
            

if __name__ == "__main__":
    infer_images('./toronto_label/01/', './ref_images/MU_clear/i.jpg', './synthetized/')

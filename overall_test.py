# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import argparse
import os
import numpy as np
import torch
import yaml

from PIL import Image
from torchvision import utils

from datasets import *
from trainer import *
from utils.functions import *
from constants import ATTR_TO_NUM

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None

from constants import DEVICE

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='new_train', help='Path to the config file.')
parser.add_argument('--attr', type=str, default='Bald,No_Beard,Smiling', help='attribute for manipulation.')
parser.add_argument('--latent_path', type=str, default='./data/celebahq_dlatents_psp.npy', help='dataset path')
parser.add_argument('--label_file', type=str, default='./data/celebahq_anno.npy', help='label file path')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='stylegan model path')
parser.add_argument('--classifier_model_path', type=str, default='./models/latent_classifier_epoch_20.pth', help='pretrained attribute classifier')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--out_path', type=str, default='./outputs/', help='output path')
opts = parser.parse_args()


log_dir = os.path.join(opts.log_path, opts.config) + '/'
config = yaml.safe_load(open('./configs/' + opts.config + '.yaml', 'r'))

save_dir = opts.out_path + 'test/scaling/'
os.makedirs(save_dir, exist_ok=True)

local_attr_dict = {attr:i for i, attr in enumerate(config["attr"].split(","))}

attributes = {'Bald': 1, 'Smiling': 1, 'Male': -1, 'No_Beard': -1}

scales = [1, 1.5, 2, 2.5, 3]

with torch.no_grad():
    for s in scales:
        attrs = config['attr'].split(',')
        attr_num = [ATTR_TO_NUM[a] for a in attrs]

        # Initialize trainer
        trainer = Trainer(config, attr_num, attrs, opts.label_file)
        trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)   
        trainer.load_model_multi(log_dir, "20_attrs")
        trainer.to(DEVICE)
        
        w_0 = np.load('./data/teaser/latent_code_00002.npy')
        w_0 = torch.tensor(w_0).to(DEVICE)
        x_0, _ = trainer.StyleGAN([w_0], input_is_latent=True, randomize_noise=False) 
        img_l = [x_0]

        coeffs = torch.zeros((1,20)).to(DEVICE)
        for i, (attr, coeff) in enumerate(attributes.items()):
            attr_idx = local_attr_dict[attr]
            coeffs[0][attr_idx] = coeff

            w_1 = trainer.T_net(w_0.view(w_0.size(0), -1), coeffs, scaling=s)
            w_1 = w_1.view(w_0.size())
            w_1 = torch.cat((w_1[:,:11,:], w_0[:,11:,:]), 1)
            x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)
            img_l.append(x_1.data)

        out = torch.cat(img_l, dim=3)
        utils.save_image(clip_img(out), save_dir + f"test_multi_{s}.png")

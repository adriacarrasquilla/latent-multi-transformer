# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import utils
from constants import ATTR_TO_NUM

from datasets import *
from trainer import *
from utils.functions import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
DEVICE = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='less_attrs', help='Path to the config file.')
parser.add_argument('--label_file', type=str, default='./data/celebahq_anno.npy', help='label file path')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='stylegan model path')
parser.add_argument('--classifier_model_path', type=str, default='./models/latent_classifier_epoch_20.pth', help='pretrained attribute classifier')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--out', type=str, default='test', help='Name of the out folder')
opts = parser.parse_args()

model = "20_attrs"
testdata_dir = './data/ffhq/'
n_steps = 11
scale = 2.0

with torch.no_grad():
    
    save_dir = os.path.join('./outputs/evaluation/' , opts.out)
    os.makedirs(save_dir, exist_ok=True)

    log_dir = os.path.join(opts.log_path, opts.config) + '/'
    config = yaml.safe_load(open('./configs/' + opts.config + '.yaml', 'r'))

    attrs = config['attr'].split(',')
    attr_num = [ATTR_TO_NUM[a] for a in attrs]

    # Initialize trainer
    trainer = Trainer(config, attr_num, attrs, opts.label_file)
    trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)   
    trainer.to(DEVICE)

    for attr in list(ATTR_TO_NUM.keys()):

        attr_num = ATTR_TO_NUM[attr]
        trainer.attr_num = ATTR_TO_NUM[attr]
        trainer.load_model(log_dir)
        
        for k in range(1000):

            w_0 = np.load(testdata_dir + 'latent_code_%05d.npy' % k)
            w_0 = torch.tensor(w_0).to(DEVICE)

            predict_lbl_0 = trainer.Latent_Classifier(w_0.view(w_0.size(0), -1))
            lbl_0 = F.sigmoid(predict_lbl_0)
            attr_pb_0 = lbl_0[:, attr_num]
            coeff = -1 if attr_pb_0 > 0.5 else 1   

            range_alpha = torch.linspace(0, scale*coeff, n_steps)
            for i,alpha in enumerate(range_alpha):
                
                w_1 = trainer.T_net(w_0.view(w_0.size(0), -1), alpha.unsqueeze(0).to(DEVICE))
                w_1 = w_1.view(w_0.size())
                w_1 = torch.cat((w_1[:,:11,:], w_0[:,11:,:]), 1)
                x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)
                utils.save_image(clip_img(x_1), save_dir + attr + '_%d'%k  + '_alpha_'+ str(i) + '.jpg')

# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import argparse
import os
import numpy as np
import torch
import yaml

from time import time

import matplotlib.pyplot as plt

from datasets import *
from trainer import *
from utils.functions import *

from PIL import Image
from constants import ATTR_TO_NUM

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
DEVICE = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='new_train', help='Path to the config file.')
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
n_samples = 5

# global setup
save_dir = os.path.join('./outputs/evaluation/' , opts.out) + '/'
os.makedirs(save_dir, exist_ok=True)

log_dir = os.path.join(opts.log_path, opts.config) + '/'
config = yaml.safe_load(open('./configs/' + opts.config + '.yaml', 'r'))

attrs = config['attr'].split(',')
attr_num = [ATTR_TO_NUM[a] for a in attrs]


def get_trainer(multi=True):
    if multi:
        trainer = Trainer(config, attr_num, attrs, opts.label_file)
        trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)
        trainer.load_model_multi(log_dir, model)
        trainer.to(DEVICE)

    return trainer

def apply_transformation(trainer, w_0, coeff, multi=True):
    if multi:
        w_1 = trainer.T_net(w_0.view(w_0.size(0), -1), coeff.unsqueeze(0).to(DEVICE))
        w_1 = w_1.view(w_0.size())
    else:
        pass
        # for i, c in enumerate(coeff):
        #     if c == 0:
        #         # skip attributes without coefficient
        #         continue
        #
        #     trainer.attr_num = ATTR_TO_NUM[attrs[i]]
        #     trainer.load_model(log_dir_single)
        #     trainer.to(DEVICE)
        #     c = torch.tensor(c).to(DEVICE)
        #
        #     w_1 = trainer.T_net(w_prev.view(w_0.size(0), -1), c.unsqueeze(0))
        #     w_1 = w_1.view(w_0.size())
        #     w_prev = w_1

    return w_1


def evaluate_scaling_vs_change_ratio(multi=True):
    with torch.no_grad():

        # Initialize trainer
        trainer = get_trainer(multi)

        all_coeffs = np.load(testdata_dir + "labels/overall.npy")

        class_rate = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))

        for k in range(n_samples):

            w_0 = np.load(testdata_dir + 'latent_code_%05d.npy' % k)
            w_0 = torch.tensor(w_0).to(DEVICE)

            predict_lbl_0 = trainer.Latent_Classifier(w_0.view(w_0.size(0), -1))
            lbl_0 = torch.sigmoid(predict_lbl_0)
            attr_pb_0 = lbl_0[:, attr_num]

            coeff = torch.tensor(all_coeffs[k]).to(DEVICE)

            scales = torch.linspace(0, scale, n_steps).to(DEVICE)
            range_coeffs = coeff * scales.reshape(-1, 1)
            for i, alpha in enumerate(range_coeffs):

                w_1 = apply_transformation(trainer=trainer, w_0=w_0, coeff=alpha, multi=multi)

                predict_lbl_1 = trainer.Latent_Classifier(w_1.view(w_0.size(0), -1))
                lbl_1 = torch.sigmoid(predict_lbl_1)
                attr_pb_1 = lbl_1[:, attr_num]

                changes = get_target_change(attr_pb_0, attr_pb_1, coeff)
                ratio = (changes.sum()/changes.size(0)).item()
                class_rate[k][i] = ratio

        rates = class_rate.mean(axis=0)
        for rate, s in zip(rates, torch.linspace(0, scale, n_steps)):
            print(f"Change rate for scale {s:.2}: {rate:.2}")

if __name__ == "__main__":
    evaluate_scaling_vs_change_ratio()

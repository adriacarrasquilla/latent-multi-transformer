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
from rich.progress import track

import matplotlib.pyplot as plt

from datasets import *
from trainer import Trainer as MultiTrainer
from original_trainer import Trainer as SingleTrainer
from utils.functions import *
from plot_evaluation import plot_ratios

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

# globals setup
model = "20_attrs"
testdata_dir = './data/ffhq/'
n_steps = 11
scale = 2.0
n_samples = 50

log_dir_single = os.path.join(opts.log_path, "original_train") + '/'

save_dir = os.path.join('./outputs/evaluation/' , opts.out) + '/'
os.makedirs(save_dir, exist_ok=True)

log_dir = os.path.join(opts.log_path, opts.config) + '/'
config = yaml.safe_load(open('./configs/' + opts.config + '.yaml', 'r'))

attrs = config['attr'].split(',')
attr_num = [ATTR_TO_NUM[a] for a in attrs]


def get_trainer(multi=True):
    if multi:
        trainer = MultiTrainer(config, attr_num, attrs, opts.label_file)
        trainer.load_model_multi(log_dir, model)
    else:
        trainer = SingleTrainer(config, None, None, opts.label_file)

    trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)
    trainer.to(DEVICE)
    return trainer


def apply_transformation(trainer, w_0, coeff, multi=True):
    # Use w_0 in case something goes wrong
    w_1 = w_0

    if multi:
        # Multi trainer prediction
        w_1 = trainer.T_net(w_0.view(w_0.size(0), -1), coeff.unsqueeze(0).to(DEVICE), scaling=1.5) # TODO: remove scaling
        w_1 = w_1.view(w_0.size())

    else:
        # Single trainer secuential prediction
        w_prev = w_0
        for i, c in enumerate(coeff):
            if c == 0:
                # skip attributes without coefficient
                continue

            trainer.attr_num = ATTR_TO_NUM[attrs[i]]
            trainer.load_model(log_dir_single)
            trainer.to(DEVICE)

            w_1 = trainer.T_net(w_prev.view(w_0.size(0), -1), c.unsqueeze(0))
            w_1 = w_1.view(w_0.size())
            w_prev = w_1

    return w_1


def evaluate_scaling_vs_change_ratio(multi=True, attr=None, attr_i=None):
    with torch.no_grad():

        # Initialize trainer
        trainer = get_trainer(multi)

        if attr and attr_i is not None:
            all_coeffs = np.load(testdata_dir + "labels/all.npy")
        else:
            all_coeffs = np.load(testdata_dir + "labels/overall.npy")

        class_rate = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))

        track_title = f"Evaluating {'multi' if multi else 'single'} model {'for ' + attr if attr else ''}"

        for k in track(range(n_samples), track_title):

            w_0 = np.load(testdata_dir + 'latent_code_%05d.npy' % k)
            w_0 = torch.tensor(w_0).to(DEVICE)

            predict_lbl_0 = trainer.Latent_Classifier(w_0.view(w_0.size(0), -1))
            lbl_0 = torch.sigmoid(predict_lbl_0)
            attr_pb_0 = lbl_0[:, attr_num]

            # Wether we are using all the coefficients or one attribute at a time
            if attr and attr_i is not None:
                coeff = torch.zeros(all_coeffs[k].shape).to(DEVICE)
                coeff[attr_i] = all_coeffs[k][attr_i]
            else:
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
        return rates

def overall_change_ratio_single_vs_multi():
    """
    Compute the overall change ratio and compare it for multi and single sequential transformers
    """

    single_rates = evaluate_scaling_vs_change_ratio(multi=False)
    multi_rates = evaluate_scaling_vs_change_ratio(multi=True)
    labels = ["Single", "Multi"]
    plot_ratios(ratios=[single_rates, multi_rates], labels=labels,
                scales=torch.linspace(0, scale, n_steps),
                output_dir=save_dir)



def individual_attr_change_ratio_single_vs_multi():
    """
    Compute the change ratio for each attribute and compare it for multi and single sequential transformers
    """

    for i, attr in enumerate(attrs):
        single_rates = evaluate_scaling_vs_change_ratio(multi=False, attr=attr, attr_i=i)
        multi_rates = evaluate_scaling_vs_change_ratio(multi=True, attr=attr, attr_i=i)
        labels = ["Single", "Multi"]
        plot_ratios(ratios=[single_rates, multi_rates], labels=labels,
                    scales=torch.linspace(0, scale, n_steps),
                    output_dir=save_dir, title=f"Target change ratio for {attr}",
                    filename=f"ratio_{attr}.png")

if __name__ == "__main__":
    individual_attr_change_ratio_single_vs_multi()

# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import argparse
import os
import gc
import numpy as np
import torch
import yaml

from time import time
from rich.progress import track

from torchvision import utils

import matplotlib.pyplot as plt

from datasets import *
from trainer import Trainer as MultiTrainer
from original_trainer import Trainer as SingleTrainer
from utils.functions import *
from plot_evaluation import plot_images, plot_nattr_evolution, plot_ratios, plot_recon_vs_reg

from PIL import Image
from constants import ATTR_TO_NUM

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
DEVICE = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="bottleneck_1", help="Path to the config file.")
parser.add_argument("--label_file", type=str, default="./data/celebahq_anno.npy", help="label file path")
parser.add_argument("--stylegan_model_path", type=str,default="./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt",help="stylegan model path")
parser.add_argument("--classifier_model_path", type=str, default="./models/latent_classifier_epoch_20.pth", help="pretrained attribute classifier")
parser.add_argument("--log_path", type=str, default="./logs/", help="log file path")
parser.add_argument("--out", type=str, default="test", help="Name of the out folder")
opts = parser.parse_args()

# globals setup
model = "20_attrs"
testdata_dir = "./data/ffhq/"
n_steps = 11
scale = 2.0

n_samples = 500

log_dir_single = os.path.join(opts.log_path, "original_train") + "/"

save_dir = os.path.join("./outputs/evaluation/", opts.out) + "/"
os.makedirs(save_dir, exist_ok=True)

log_dir = os.path.join(opts.log_path, opts.config) + "/"
config = yaml.safe_load(open("./configs/" + opts.config + ".yaml", "r"))

attrs = config["attr"].split(",")
attr_num = [ATTR_TO_NUM[a] for a in attrs]

from evaluation import get_trainer, apply_transformation, get_ratios_from_sample


def evaluate_scaling_vs_change_ratio(config_name, attr=None, attr_i=None, orders=None, n_samples=n_samples,
                                     n_steps=n_steps, scale=scale):

    config = yaml.safe_load(open("./configs/" + config_name + ".yaml", "r"))
    log_dir = os.path.join(opts.log_path, config_name) + "/"

    with torch.no_grad():
        # Initialize trainer
        trainer = get_trainer(True, config=config, log_dir=log_dir, attr_num=attr_num, attrs=attrs)

        if (attr and attr_i is not None) or orders is not None:
            all_coeffs = np.load(testdata_dir + "labels/all.npy")
            extra = f'for {attr}' if orders is None else f'for {orders.shape[1]} attrs'
        else:
            all_coeffs = np.load(testdata_dir + "labels/overall.npy")
            extra = ""

        class_ratios = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))
        ident_ratios = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))
        attr_ratios = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))

        track_title = f"Evaluating multi model " + extra

        for k in track(range(n_samples), track_title):
            w_0 = np.load(testdata_dir + "latent_code_%05d.npy" % k)
            w_0 = torch.tensor(w_0).to(DEVICE)

            predict_lbl_0 = trainer.Latent_Classifier(w_0.view(w_0.size(0), -1))
            lbl_0 = torch.sigmoid(predict_lbl_0)
            attr_pb_0 = lbl_0[:, attr_num]

            # Wether we are using all the coefficients or one attribute at a time
            if attr and attr_i is not None:
                coeff = torch.zeros(len(attrs)).to(DEVICE)
                coeff[attr_i] = all_coeffs[k][attr_i]
            elif orders is not None:
                coeff = torch.zeros(len(attrs)).to(DEVICE)
                coeff[orders[k]] = torch.tensor(all_coeffs[k][orders[k]], dtype=torch.float).to(DEVICE)
            else:
                coeff = torch.tensor(all_coeffs[k]).to(DEVICE)

            scales = torch.linspace(0, scale, n_steps).to(DEVICE)
            range_coeffs = coeff * scales.reshape(-1, 1)
            for i, alpha in enumerate(range_coeffs):
                w_1 = apply_transformation(
                    trainer=trainer, w_0=w_0, coeff=alpha, multi=True
                )

                predict_lbl_1 = trainer.Latent_Classifier(w_1.view(w_0.size(0), -1))
                lbl_1 = torch.sigmoid(predict_lbl_1)
                attr_pb_1 = lbl_1[:, attr_num]

                ident_ratio = trainer.MSEloss(w_1, w_0)
                attr_ratio = get_attr_change(lbl_0, lbl_1, coeff, attr_num)
                class_ratio = get_target_change(attr_pb_0, attr_pb_1, coeff)

                class_ratios[k][i] = class_ratio
                ident_ratios[k][i] = ident_ratio
                attr_ratios[k][i] = attr_ratio

        class_r = class_ratios.mean(axis=0)
        recons = ident_ratios.mean(axis=0)
        attr_r = attr_ratios.mean(axis=0)

        return class_r, recons, attr_r


def overall_change_ratio_single_vs_multi():
    """
    Compute the overall change ratio and compare it for multi and single sequential transformers
    """

    multi_rates, _, _ = evaluate_scaling_vs_change_ratio(config_name=opts.config)
    labels = ["Single", "Multi"]
    plot_ratios(
        ratios=[multi_rates, multi_rates],
        labels=labels,
        scales=torch.linspace(0, scale, n_steps),
        output_dir=save_dir,
    )


def individual_attr_change_ratio_single_vs_multi():
    """
    Compute the change ratio for each attribute and compare it for multi and single sequential transformers
    """

    for i, attr in enumerate(attrs[:]):

        single_rates, single_recons, single_regs = evaluate_scaling_vs_change_ratio(multi=False, attr=attr, attr_i=i)
        multi_rates, multi_recons, multi_regs = evaluate_scaling_vs_change_ratio(multi=True, attr=attr, attr_i=i)
        labels = ["Single", "Multi"]

        plot_ratios(
            ratios=[single_rates, multi_rates],
            labels=labels,
            scales=torch.linspace(0, scale, n_steps),
            output_dir=save_dir + "indiv_ratio/",
            title=f"Target change ratio for {attr}",
            filename=f"ratio_{attr}.png",
        )
        plot_recon_vs_reg(
            ratios=[single_rates, multi_rates],
            recons=[single_recons, multi_recons],
            regs=[single_regs, multi_regs],
            labels=labels,
            scales=torch.linspace(0, scale, n_steps),
            output_dir=save_dir + "indiv_recon_vs_reg/",
            title=f"Target change ratio for {attr}",
            filename=f"recon_vs_reg_{attr}.png",
        )

def different_nattrs_ratio_single_vs_multi():
    orders = np.load(testdata_dir + "labels/attr_order.npy")
    all_rates = np.zeros((len(attrs), 2, len(torch.linspace(0, scale, n_steps))))
    all_recons = np.zeros((len(attrs), 2, len(torch.linspace(0, scale, n_steps))))
    all_regs = np.zeros((len(attrs), 2, len(torch.linspace(0, scale, n_steps))))
    for i in range(len(attrs[:])):
        single_rates, single_recons, single_regs = evaluate_scaling_vs_change_ratio(multi=False, orders=orders[:, :i+1])
        multi_rates, multi_recons, multi_regs = evaluate_scaling_vs_change_ratio(multi=True, orders=orders[:, :i+1])
        labels = ["Single", "Multi"]

        plot_recon_vs_reg(
            ratios=[single_rates, multi_rates],
            recons=[single_recons, multi_recons],
            regs=[single_regs, multi_regs],
            labels=labels,
            scales=torch.linspace(0, scale, n_steps),
            output_dir=save_dir + "n_attrs/",
            title=f"Target change ratio for {i+1} attrs",
            filename=f"{i+1}_attrs.png",
        )

        # Store results to avoid repeating computation
        all_rates[i][0] = single_rates
        all_rates[i][1] = multi_rates

        all_recons[i][0] = single_recons
        all_recons[i][1] = multi_recons

        all_regs[i][0] = single_regs
        all_regs[i][1] = multi_regs

        np.save(save_dir + "n_attrs/all_rates.npy", all_rates)
        np.save(save_dir + "n_attrs/all_recons.npy", all_recons)
        np.save(save_dir + "n_attrs/all_regs.npy", all_regs)


def nattrs_ratio_progression_single_vs_multi():
    all_rates = np.load(save_dir + "n_attrs/all_rates.npy")
    all_recons = np.load(save_dir + "n_attrs/all_recons.npy")
    all_regs = np.load(save_dir + "n_attrs/all_regs.npy")

    labels = ["Single", "Multi"]
    coeffs = torch.linspace(0, scale, n_steps)

    for n in [2,5,8,10]:
        plot_nattr_evolution(
            ratios=[all_rates[:,0, n], all_rates[:, 1, n]],
            recons=[all_recons[:, 0, n], all_recons[:, 1, n]],
            regs=[all_regs[:, 0, n], all_regs[:, 1, n]],
            labels=labels,
            output_dir=save_dir + "n_attrs_evolution/",
            title=f"Change ratio over n attrs, scaling = {coeffs[n]:.2}",
            filename=f"n_attrs_evolution_{coeffs[n]:.2}.png",
        )


if __name__ == "__main__":
    overall_change_ratio_single_vs_multi()
    # individual_attr_change_ratio_single_vs_multi()
    # different_nattrs_ratio_single_vs_multi()
    # nattrs_ratio_progression_single_vs_multi()

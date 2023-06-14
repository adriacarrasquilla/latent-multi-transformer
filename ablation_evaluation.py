# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import argparse
import os
import numpy as np
import torch
import yaml

from rich.progress import track

from datasets import *
from utils.functions import *
from plot_evaluation import plot_ratios

from evaluation import get_trainer, apply_transformation


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
parser.add_argument("--stylegan_model_path",type=str,default="./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt",help="stylegan model path")
parser.add_argument("--classifier_model_path",type=str,default="./models/latent_classifier_epoch_20.pth",help="pretrained attribute classifier")
parser.add_argument("--log_path", type=str, default="./logs/", help="log file path")
parser.add_argument("--out", type=str, default="ablation", help="Name of the out folder")
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

coeff_map = [0, 4, 2]


def evaluate_scaling_vs_change_ratio_bottleneck(
    config_name,
    attr=None,
    attr_i=None,
    orders=None,
    n_samples=n_samples,
    n_steps=n_steps,
    scale=scale,
):

    config = yaml.safe_load(open("./configs/" + config_name + ".yaml", "r"))
    log_dir = os.path.join(opts.log_path, config_name) + "/"

    with torch.no_grad():
        # Initialize trainer
        trainer = get_trainer(
            True, config=config, log_dir=log_dir, attr_num=attr_num, attrs=attrs
        )

        if (attr and attr_i is not None) or orders is not None:
            all_coeffs = np.load(testdata_dir + "labels/all.npy")
            extra = f"for {attr}" if orders is None else f"for {orders.shape[1]} attrs"
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
                less_orders = [o for o in orders[k] if o in coeff_map]
                coeff = torch.tensor(all_coeffs[k][coeff_map], dtype=torch.float).to(
                    DEVICE
                )
            else:
                coeff = torch.tensor(all_coeffs[k][coeff_map]).to(DEVICE)

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


def overall_change_ratio_bottleneck():
    """
    Compute the overall change ratio and compare it for multi and single sequential transformers
    """

    # orders = None
    orders = np.load(testdata_dir + "labels/attr_order.npy")
    configs = [f"bottleneck_{i}" for i in ["1", "3", "6"]]
    rates = []

    for conf in configs:
        multi_rates, _, _ = evaluate_scaling_vs_change_ratio_bottleneck(
            config_name=conf, orders=orders, n_samples=300
        )
        rates.append(multi_rates)

    plot_ratios(
        ratios=rates,
        labels=[f"{i}" for i in [512, 512*3, 512*6]],
        scales=torch.linspace(0, scale, n_steps),
        output_dir=save_dir,
        title="Target Change Ratio for different w sizes",
        filename="ablation_size.png"
    )


if __name__ == "__main__":
    overall_change_ratio_bottleneck()

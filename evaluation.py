# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import argparse
import os
import numpy as np
import torch
import yaml

from rich.progress import track

from datasets import *
from trainer import Trainer as MultiTrainer
from original_trainer import Trainer as SingleTrainer
from utils.functions import *
from plot_evaluation import plot_nattr_evolution_fixed_coeff, plot_nattr_evolution_fixed_attr, plot_ratios, plot_ratios_stacked, plot_recon_vs_reg, plot_recon_vs_reg_row

from PIL import Image
from constants import ATTR_TO_NUM

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
DEVICE = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="new_train", help="Path to the config file.")
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

n_samples = 300

log_dir_single = os.path.join(opts.log_path, "original_train") + "/"

save_dir = os.path.join("./outputs/evaluation/", opts.out) + "/"
os.makedirs(save_dir, exist_ok=True)

log_dir = os.path.join(opts.log_path, opts.config) + "/"
config = yaml.safe_load(open("./configs/" + opts.config + ".yaml", "r"))

attrs = config["attr"].split(",")
attr_num = [ATTR_TO_NUM[a] for a in attrs]


def get_trainer(multi=True, config=config, log_dir=log_dir, attr_num=attr_num, attrs=attrs):
    if multi:
        trainer = MultiTrainer(config, attr_num, attrs, opts.label_file)
        trainer.load_model_multi(log_dir, model)
    else:
        trainer = SingleTrainer(config, None, None, opts.label_file)

    trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)
    trainer.to(DEVICE)
    return trainer


def apply_transformation(trainer, w_0, coeff, attrs=attrs, multi=True):
    # Use w_0 in case something goes wrong
    w_1 = w_0

    if multi:
        # Multi trainer prediction
        w_1 = trainer.T_net(
            w_0.view(w_0.size(0), -1), coeff.unsqueeze(0).to(DEVICE), scaling=1.5
        )
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


def get_ratios_from_sample(w_0, w_1, coeff, trainer):
    predict_lbl_0 = trainer.Latent_Classifier(w_0.view(w_0.size(0), -1))
    lbl_0 = torch.sigmoid(predict_lbl_0)
    attr_pb_0 = lbl_0[:, attr_num]

    predict_lbl_1 = trainer.Latent_Classifier(w_1.view(w_0.size(0), -1))
    lbl_1 = torch.sigmoid(predict_lbl_1)
    attr_pb_1 = lbl_1[:, attr_num]

    ratio = get_target_change(attr_pb_0, attr_pb_1, coeff, mean=False)

    return ratio


def evaluate_scaling_vs_change_ratio(multi=True, attr=None, attr_i=None, orders=None, n_samples=n_samples,
                                     return_mean=True, n_steps=n_steps, scale=scale):
    with torch.no_grad():
        # Initialize trainer
        trainer = get_trainer(multi)

        if (attr and attr_i is not None) or orders is not None:
            all_coeffs = np.load(testdata_dir + "labels/all.npy")
            extra = f'for {attr}' if orders is None else f'for {orders.shape[1]} attrs'
        else:
            all_coeffs = np.load(testdata_dir + "labels/overall.npy")
            extra = ""

        class_ratios = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))
        ident_ratios = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))
        attr_ratios = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))

        track_title = f"Evaluating {'multi' if multi else 'single'} model " + extra

        for k in track(range(n_samples), track_title):
            w_0 = np.load(testdata_dir + "latent_code_%05d.npy" % k)
            w_0 = torch.tensor(w_0).to(DEVICE)

            predict_lbl_0 = trainer.Latent_Classifier(w_0.view(w_0.size(0), -1))
            lbl_0 = torch.sigmoid(predict_lbl_0)
            attr_pb_0 = lbl_0[:, attr_num]

            # Wether we are using all the coefficients or one attribute at a time
            if attr and attr_i is not None:
                coeff = torch.zeros(all_coeffs[k].shape).to(DEVICE)
                coeff[attr_i] = all_coeffs[k][attr_i]
            elif orders is not None:
                coeff = torch.zeros(all_coeffs[k].shape).to(DEVICE)
                coeff[orders[k]] = torch.tensor(all_coeffs[k][orders[k]], dtype=torch.float).to(DEVICE)
            else:
                coeff = torch.tensor(all_coeffs[k]).to(DEVICE)

            scales = torch.linspace(0, scale, n_steps).to(DEVICE)
            range_coeffs = coeff * scales.reshape(-1, 1)
            for i, alpha in enumerate(range_coeffs):
                w_1 = apply_transformation(
                    trainer=trainer, w_0=w_0, coeff=alpha, multi=multi
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

        if return_mean:
            class_r = class_ratios.mean(axis=0)
            recons = ident_ratios.mean(axis=0)
            attr_r = attr_ratios.mean(axis=0)
        else:
            class_r = class_ratios
            recons = ident_ratios
            attr_r = attr_ratios

        return class_r, recons, attr_r


def overall_change_ratio_single_vs_multi():
    """
    Compute the overall change ratio and compare it for multi and single sequential transformers
    """

    n_samples = 999
    single_rates, _, _ = evaluate_scaling_vs_change_ratio(multi=False, n_samples=n_samples)
    multi_rates, _, _ = evaluate_scaling_vs_change_ratio(multi=True, n_samples=n_samples)
    labels = ["Single", "Multi"]
    plot_ratios(
        ratios=[single_rates, multi_rates],
        labels=labels,
        scales=torch.linspace(0, scale, n_steps),
        output_dir=save_dir,
        figsize=(5,3)
    )


def individual_attr_change_ratio_single_vs_multi():
    """
    Compute the change ratio for each attribute and compare it for multi and single sequential transformers
    """

    all_single = []
    all_multi = []
    sel_ratios = []
    sel_regs = []
    sel_recons = []
    labels = ["Single", "Multi"]
    attr_selected = ["Attractive", "Arched_Eyebrows", "Young", "Gray_Hair"]
    for i, attr in enumerate(attrs[:]):

        single_rates, single_recons, single_regs = evaluate_scaling_vs_change_ratio(multi=False, attr=attr, attr_i=i)
        multi_rates, multi_recons, multi_regs = evaluate_scaling_vs_change_ratio(multi=True, attr=attr, attr_i=i)

        all_single.append(single_rates)
        all_multi.append(multi_rates)

        if attr in attr_selected:
            sel_ratios.append([single_rates, multi_rates])
            sel_recons.append([single_recons, multi_recons])
            sel_regs.append([single_regs, multi_regs])

    plot_ratios_stacked(
        single_ratios=all_single,
        multi_ratios=all_multi,
        attrs=attrs,
        scales=torch.linspace(0, scale, n_steps),
        output_dir=save_dir + "indiv_ratio/",
    )
    plot_recon_vs_reg_row(all_regs=sel_regs, all_ratios=sel_ratios, all_recons=sel_recons, labels=labels, titles=attr_selected, filename="individual_ratio_vs_scores.png")


def different_nattrs_ratio_single_vs_multi(compute=False):
    orders = np.load(testdata_dir + "labels/attr_order.npy")
    labels = ["Single", "Multi"]
    if compute:
        all_rates = np.zeros((len(attrs), 2, len(torch.linspace(0, scale, n_steps))))
        all_recons = np.zeros((len(attrs), 2, len(torch.linspace(0, scale, n_steps))))
        all_regs = np.zeros((len(attrs), 2, len(torch.linspace(0, scale, n_steps))))

        for i in range(len(attrs[:])):
            single_rates, single_recons, single_regs = evaluate_scaling_vs_change_ratio(multi=False, orders=orders[:, :i+1])
            multi_rates, multi_recons, multi_regs = evaluate_scaling_vs_change_ratio(multi=True, orders=orders[:, :i+1])

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
    else:
        all_rates = np.load(save_dir + "n_attrs/all_rates.npy")
        all_recons = np.load(save_dir + "n_attrs/all_recons.npy")
        all_regs = np.load(save_dir + "n_attrs/all_regs.npy")

    plot_nattr_evolution_fixed_attr(
        ratios=all_rates,
        recons=all_recons,
        regs=all_regs,
        labels=labels,
        n_attrs=[0,4,12,19],
        output_dir=save_dir + "n_attrs/",
        filename=f"fixed_n.png",
    )



def nattrs_ratio_progression_single_vs_multi():
    all_rates = np.load(save_dir + "n_attrs/all_rates.npy")
    all_recons = np.load(save_dir + "n_attrs/all_recons.npy")
    all_regs = np.load(save_dir + "n_attrs/all_regs.npy")

    labels = ["Single", "Multi"]

    coeffs = torch.linspace(0, scale, n_steps)

    plot_nattr_evolution_fixed_coeff(
        ratios=all_rates,
        recons=all_recons,
        regs=all_regs,
        coeffs=coeffs,
        n_attrs=[2,5,8,10],
        labels=labels,
        output_dir=save_dir + "n_attrs_evolution/",
        filename="fixed_coeff.png",
    )


if __name__ == "__main__":
    overall_change_ratio_single_vs_multi()
    # individual_attr_change_ratio_single_vs_multi()
    # different_nattrs_ratio_single_vs_multi(compute=False)
    # nattrs_ratio_progression_single_vs_multi()
    # quality_eval()

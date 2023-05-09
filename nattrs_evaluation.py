from rich.progress import track
import torch
import numpy as np
import os

import yaml

from constants import ATTR_TO_NUM, DEVICE, NATTRS_NUM_TO_IDX, NUM_TO_ATTR
from plot_evaluation import plot_ratios
from evaluation import apply_transformation, get_trainer
from utils.functions import clip_img, get_attr_change, get_target_change
from torchvision import utils

log_dir_single = os.path.join("./logs", "original_train") + "/"
save_dir = os.path.join("./outputs/evaluation/n_attrs") + "/"
os.makedirs(save_dir, exist_ok=True)

model = "20_attrs"
testdata_dir = "./data/ffhq/"
n_steps = 11
scale = 2.0

n_samples = 500


def evaluate_scaling_vs_change_ratio_nattrs(
        attr_num, config, attrs, log_dir, multi=True, attr=None, attr_i=None, orders=None, n_samples=n_samples,
        return_mean=True, n_steps=n_steps, scale=scale, conf_name="",
    ):

    with torch.no_grad():
        # Initialize trainer
        trainer = get_trainer(multi, config=config, attr_num=attr_num, attrs=attrs, log_dir=log_dir)

        all_coeffs = np.load(testdata_dir + f"labels/{conf_name}.npy")

        class_ratios = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))
        ident_ratios = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))
        attr_ratios = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))

        track_title = f"Evaluating {'multi' if multi else 'single'} model"

        # attr_ids = [NATTRS_NUM_TO_IDX[NUM_TO_ATTR[a]] for a in attr_num]

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
                if k % 100 == 0:
                    w_1 = torch.cat((w_1[:,:11,:], w_0[:,11:,:]), 1)
                    x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)
                    utils.save_image(clip_img(x_1), save_dir + conf_name + ("_multi_" if multi else "_single_") + str(k) + '.jpg')

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


def n_attrs_evaluation():
    """
    Compute the overall change ratio and compare it for multi and single sequential transformers
    """
    config_paths = ["5_attrs_nc", "5_attrs_c", "10_attrs", "15_attrs"]

    for c_path in config_paths:
        config = yaml.safe_load(open("./configs/" + c_path + ".yaml", "r"))
        attrs = config["attr"].split(",")
        attr_num = [ATTR_TO_NUM[a] for a in attrs]
        log_dir = f"./logs/{c_path}"

        multi_rates, _, _ = evaluate_scaling_vs_change_ratio_nattrs(attr_num=attr_num, config=config, attrs=attrs, log_dir=log_dir, multi=True, conf_name=c_path)
        single_rates, _, _ = evaluate_scaling_vs_change_ratio_nattrs(attr_num=attr_num, config=config,attrs=attrs, log_dir=log_dir, multi=False, conf_name=c_path)
        labels = ["Single", "Multi"]
        plot_ratios(
            ratios=[single_rates, multi_rates],
            labels=labels,
            scales=torch.linspace(0, scale, n_steps),
            output_dir=save_dir,
            filename=f"{c_path}_ratio.png"
        )

if __name__ == "__main__":
    n_attrs_evaluation()

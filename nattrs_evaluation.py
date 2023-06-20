from rich.progress import track
import torch
import numpy as np
import os

import yaml

from constants import ATTR_TO_NUM, DEVICE
from plot_evaluation import plot_ratios_row, plot_recon_vs_reg_row
from evaluation import apply_transformation, get_trainer
from utils.functions import get_attr_change, get_target_change

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
        attr_ratios_corr = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))
        attr_ratios_uncorr = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))

        track_title = f"Evaluating {'multi' if multi else 'single'} model"

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

            corr_vector = trainer.get_correlation_multi(attr_num, coeffs=coeff)
            correlated_mask = (0 < corr_vector) & ( corr_vector < 0.7)
            uncorrelated_mask = corr_vector >= 0.7

            scales = torch.linspace(0, scale, n_steps).to(DEVICE)
            range_coeffs = coeff * scales.reshape(-1, 1)
            for i, alpha in enumerate(range_coeffs):
                w_1 = apply_transformation(
                    trainer=trainer, w_0=w_0, coeff=alpha, multi=multi, attrs=attrs
                )

                predict_lbl_1 = trainer.Latent_Classifier(w_1.view(w_0.size(0), -1))
                lbl_1 = torch.sigmoid(predict_lbl_1)
                attr_pb_1 = lbl_1[:, attr_num]

                ident_ratio = trainer.MSEloss(w_1, w_0)
                attr_ratio = get_attr_change(lbl_0, lbl_1, coeff, attr_num)
                attr_ratios_all, target_mask = get_attr_change(lbl_0, lbl_1, coeff, attr_num, mean=False)
                class_ratio = get_target_change(attr_pb_0, attr_pb_1, coeff)

                current_correlated_mask = correlated_mask[target_mask]
                current_uncorrelated_mask = uncorrelated_mask[target_mask]

                class_ratios[k][i] = class_ratio
                ident_ratios[k][i] = ident_ratio
                attr_ratios[k][i] = attr_ratio
                if current_correlated_mask.any():
                    attr_ratios_corr[k][i] = (attr_ratios_all[current_correlated_mask].sum() / attr_ratios_all[current_correlated_mask].size(0))
                else:
                    attr_ratios_corr[k][i] = 1

                if current_uncorrelated_mask.any():
                    attr_ratios_uncorr[k][i] = (attr_ratios_all[current_uncorrelated_mask].sum() / attr_ratios_all[current_uncorrelated_mask].size(0))
                else:
                    attr_ratios_corr[k][i] = 1

        if return_mean:
            class_r = class_ratios.mean(axis=0)
            recons = ident_ratios.mean(axis=0)
            attr_r = attr_ratios.mean(axis=0)
            attr_r_corr = attr_ratios_corr.mean(axis=0)
            attr_r_uncorr = attr_ratios_uncorr.mean(axis=0)
        else:
            class_r = class_ratios
            recons = ident_ratios
            attr_r = attr_ratios
            attr_r_corr = attr_ratios_corr
            attr_r_uncorr = attr_ratios_uncorr

        return class_r, recons, attr_r, attr_r_corr, attr_r_uncorr


def n_attrs_evaluation(compute=False):
    """
    Compute the overall change ratio and compare it for multi and single sequential transformers
    """
    config_paths = ["5_attrs_nc", "5_attrs_c", "10_attrs", "15_attrs"]
    all_ratios = []
    all_recons = []
    all_regs = []
    titles = ["5 correlated attributes", "5 non-correlated attributes", "10 attributes", "15 attributes"]
    labels = ["Single", "Multi"]
    n_samples = 100

    if compute:
        for c_path in config_paths:
            config = yaml.safe_load(open("./configs/" + c_path + ".yaml", "r"))
            attrs = config["attr"].split(",")
            attr_num = [ATTR_TO_NUM[a] for a in attrs]
            log_dir = f"./logs/{c_path}"

            multi_rates, multi_recons, multi_regs, multi_corr, multi_uncorr = evaluate_scaling_vs_change_ratio_nattrs(attr_num=attr_num, config=config, attrs=attrs, log_dir=log_dir, multi=True, conf_name=c_path, n_samples=n_samples)
            single_rates, single_recons, single_regs, single_corr, single_uncorr = evaluate_scaling_vs_change_ratio_nattrs(attr_num=attr_num, config=config,attrs=attrs, log_dir=log_dir, multi=False, conf_name=c_path, n_samples=n_samples)

            all_ratios.append([single_rates, multi_rates])
            all_recons.append([single_recons, multi_recons])
            all_regs.append([single_regs, multi_regs])

        all_ratios = np.array(all_ratios)
        all_recons = np.array(all_recons)
        all_regs = np.array(all_regs)

        np.save("./outputs/evaluation/all_ratios.npy", all_ratios)
        np.save("./outputs/evaluation/all_recons.npy", all_recons)
        np.save("./outputs/evaluation/all_regs.npy", all_regs)

    else:
        all_ratios = np.load("./outputs/evaluation/all_ratios.npy")
        all_recons = np.load("./outputs/evaluation/all_recons.npy")
        all_regs = np.load("./outputs/evaluation/all_regs.npy")


    plot_ratios_row(all_ratios=all_ratios, labels=labels, scales=torch.linspace(0, scale, n_steps), titles=titles, filename="ratios_n_attrs.png")
    plot_recon_vs_reg_row(all_regs=all_regs, all_recons=all_recons, all_ratios=all_ratios, titles=titles, labels=labels, filename="n_attrs.png")



if __name__ == "__main__":
    n_attrs_evaluation(compute=False)

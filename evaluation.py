# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import torch
import yaml
from rich.progress import track

import random

random.seed(1)

from PIL import Image

from datasets import *
from original_trainer import Trainer as SingleTrainer
from trainer import Trainer as MultiTrainer
from utils.functions import *
from torchvision import utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
DEVICE = torch.device('cuda')

from constants import ATTR_TO_NUM


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='new_train', help='Path to the config file.')
parser.add_argument('--label_file', type=str, default='./data/celebahq_anno.npy', help='label file path')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='stylegan model path')
parser.add_argument('--classifier_model_path', type=str, default='./models/latent_classifier_epoch_20.pth', help='pretrained attribute classifier')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
opts = parser.parse_args()

config = yaml.safe_load(open('./configs/' + opts.config + '.yaml', 'r'))
exp = opts.config
os.makedirs(f"./out_images/{exp}", exist_ok=True)

# Load input latent codes
testdata_dir = './data/ffhq/'
n_steps = 5
scale = 2.0

save_dir = './outputs/evaluation/new_train'
log_dir_single = os.path.join(opts.log_path, "original_train") + '/'

os.makedirs(save_dir, exist_ok=True)

log_dir = os.path.join(opts.log_path, opts.config) + '/'

n_attrs = 20
attrs = config['attr'].split(',')
attr_num = [ATTR_TO_NUM[a] for a in attrs]
model = f"{n_attrs}_attrs"

n_samples = 1000


def compute_sequential_loss(w, w_1, attr_nums, coeff, trainer):
    w_0 = w
    predict_lbl_0 = trainer.Latent_Classifier(w_0.view(w.size(0), -1))
    lbl_0 = torch.sigmoid(predict_lbl_0)

    attr_pb_0 = lbl_0[torch.arange(lbl_0.shape[0]), attr_nums]

    coeff_idx = coeff.nonzero(as_tuple=True)[0].tolist()

    target_pb = torch.clamp(attr_pb_0 + coeff, 0, 1).round()
    target_pb = target_pb[coeff_idx]

    predict_lbl_1 = trainer.Latent_Classifier(w_1.view(w.size(0), -1))

    # Pb loss
    T_coeff = target_pb.size(0)/(target_pb.sum(0) + 1e-8)
    F_coeff = target_pb.size(0)/(target_pb.size(0) - target_pb.sum(0) + 1e-8)
    mask_pb = T_coeff.float() * target_pb + F_coeff.float() * (1-target_pb)

    pred_pb = predict_lbl_1[torch.arange(predict_lbl_1.shape[0]), attr_nums]
    loss_pb = trainer.BCEloss(pred_pb[coeff_idx], target_pb, reduction='none')*mask_pb

    loss_pb = loss_pb.mean()

    # Latent code recon
    loss_recon = trainer.MSEloss(w_1, w_0)

    # Reg loss
    mask = torch.tensor(trainer.get_correlation_multi(attr_nums, threshold=1, coeffs=coeff)).type_as(predict_lbl_0)
    # mask = mask.repeat(predict_lbl_0.size(0), 1)
    loss_reg = trainer.MSEloss(predict_lbl_1*mask, predict_lbl_0*mask)
    
    # Total loss
    w_recon, w_pb, w_reg = config['w']['recon'], config['w']['pb'], config['w']['reg']
    loss =  w_pb * loss_pb + w_recon*loss_recon + w_reg * loss_reg

    return loss, w_pb * loss_pb, w_reg * loss_reg, w_recon * loss_recon


def eval_multi(save_img=True, scaling=1):
    with torch.no_grad():
        
        # Initialize trainer
        trainer = MultiTrainer(config, attr_num, attrs, opts.label_file)
        trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)   
        trainer.load_model_multi(log_dir, model)
        trainer.to(DEVICE)

        coeffs = np.load(testdata_dir + "labels/overall.npy")

        losses = torch.zeros((n_samples, 4)).to(DEVICE)

        for k in track(range(n_samples), "Evaluating Multi Attribute model..."):

            w_0 = np.load(testdata_dir + 'latent_code_%05d.npy' % k)
            w_0 = torch.tensor(w_0).to(DEVICE)

            coeff = torch.tensor(coeffs[k]).to(DEVICE)

            w_1 = trainer.T_net(w_0.view(w_0.size(0), -1), coeff.unsqueeze(0), scaling=scaling)
            w_1 = w_1.view(w_0.size())

            loss, loss_pb, loss_reg, loss_recon = compute_sequential_loss(w_0, w_1, attr_num, coeff, trainer)

            losses[k] = torch.tensor([loss.item(), loss_pb.item(), loss_reg, loss_recon]).to(DEVICE)

            if k % 100 == 0 and save_img:
                w_1 = torch.cat((w_1[:,:11,:], w_0[:,11:,:]), 1)
                x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)
                utils.save_image(clip_img(x_1), save_dir + "multi_" + str(k) + '.jpg')

        return losses.mean(dim=0).detach().cpu().numpy()


def eval_multi_n(n=1, scaling=1):
    with torch.no_grad():
        
        # Initialize trainer
        trainer = MultiTrainer(config, attr_num, attrs, opts.label_file)
        trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)   
        trainer.load_model_multi(log_dir, model)
        trainer.to(DEVICE)

        coeffs = np.load(testdata_dir + "labels/all.npy")

        losses = torch.zeros((n_samples, 4)).to(DEVICE)

        for k in track(range(n_samples), f"Evaluating Multi model for {n} attributes, scaling = {scaling}..."):
            local_attrs = random.sample(range(coeffs.shape[1]), random.randint(1,n))

            w_0 = np.load(testdata_dir + 'latent_code_%05d.npy' % k)
            w_0 = torch.tensor(w_0).to(DEVICE)

            coeff = np.zeros_like(coeffs[k])
            coeff[local_attrs] = coeffs[k][local_attrs]
            coeff = torch.tensor(coeff).to(DEVICE)

            w_1 = trainer.T_net(w_0.view(w_0.size(0), -1), coeff.unsqueeze(0), scaling=scaling)
            w_1 = w_1.view(w_0.size())

            loss, loss_pb, loss_reg, loss_recon = compute_sequential_loss(w_0, w_1, attr_num, coeff, trainer)

            losses[k] = torch.tensor([loss.item(), loss_pb.item(), loss_reg, loss_recon]).to(DEVICE)

            if k == 500:
                w_1 = torch.cat((w_1[:,:11,:], w_0[:,11:,:]), 1)
                x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)
                utils.save_image(clip_img(x_1), save_dir + f"multi_{n}_attrs_" + str(k) + '.jpg')

        return losses.mean(dim=0).detach().cpu().numpy()

def eval_single(save_img=True):
    with torch.no_grad():
        
        # Initialize trainer
        trainer = SingleTrainer(config, None, None, opts.label_file)
        trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)   

        coeffs = np.load(testdata_dir + "labels/overall.npy")

        losses = torch.zeros((n_samples, 4)).to(DEVICE)

        for k in track(range(n_samples), "Evaluating Single Attribute models..."):

            w_0 = np.load(testdata_dir + 'latent_code_%05d.npy' % k)
            w_0 = torch.tensor(w_0).to(DEVICE)
            w_1 = w_0
            w_prev = w_0

            coeff = coeffs[k]

            for i, c in enumerate(coeff):

                if c == 0:
                    # skip attributes without coefficient
                    continue

                trainer.attr_num = ATTR_TO_NUM[attrs[i]]
                trainer.load_model(log_dir_single)
                trainer.to(DEVICE)
                c = torch.tensor(c).to(DEVICE)

                w_1 = trainer.T_net(w_prev.view(w_0.size(0), -1), c.unsqueeze(0))
                w_1 = w_1.view(w_0.size())
                w_prev = w_1

            coeff = torch.tensor(coeff).to(DEVICE)

            loss, loss_pb, loss_reg, loss_recon = compute_sequential_loss(w_0, w_1, attr_num, coeff, trainer)

            losses[k] = torch.tensor([loss.item(), loss_pb.item(), loss_reg, loss_recon]).to(DEVICE)

            if k % 100 == 0 and save_img:
                w_1 = torch.cat((w_1[:,:11,:], w_0[:,11:,:]), 1)
                x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)
                utils.save_image(clip_img(x_1), save_dir + "single_" + str(k) + '.jpg')

        return losses.mean(dim=0).detach().cpu().numpy()


def eval_single_n(n=1):
    with torch.no_grad():
        
        # Initialize trainer
        trainer = SingleTrainer(config, None, None, opts.label_file)
        trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)   
        trainer.to(DEVICE)

        coeffs = np.load(testdata_dir + "labels/all.npy")

        losses = torch.zeros((n_samples, 4)).to(DEVICE)

        for k in track(range(n_samples), f"Evaluating Single for {n} attributes..."):
            local_attrs = random.sample(range(coeffs.shape[1]), random.randint(1,n))

            coeff = np.zeros_like(coeffs[k])
            coeff[local_attrs] = coeffs[k][local_attrs]

            w_0 = np.load(testdata_dir + 'latent_code_%05d.npy' % k)
            w_0 = torch.tensor(w_0).to(DEVICE)
            w_1 = w_0
            w_prev = w_0

            for i, c in enumerate(coeff):

                if c == 0:
                    # skip attributes without coefficient
                    continue

                trainer.attr_num = ATTR_TO_NUM[attrs[i]]
                trainer.load_model(log_dir_single)
                trainer.to(DEVICE)
                c = torch.tensor(c).to(DEVICE)

                w_1 = trainer.T_net(w_prev.view(w_0.size(0), -1), c.unsqueeze(0))
                w_1 = w_1.view(w_0.size())
                w_prev = w_1

            coeff = torch.tensor(coeff).to(DEVICE)

            loss, loss_pb, loss_reg, loss_recon = compute_sequential_loss(w_0, w_1, attr_num, coeff, trainer)

            losses[k] = torch.tensor([loss.item(), loss_pb.item(), loss_reg, loss_recon]).to(DEVICE)

            if k == 500:
                w_1 = torch.cat((w_1[:,:11,:], w_0[:,11:,:]), 1)
                x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)
                utils.save_image(clip_img(x_1), save_dir + f"single_{n}_attrs_" + str(k) + '.jpg')

        return losses.mean(dim=0).detach().cpu().numpy()


def all_n_experiment(suffix="", multi=True, single=True, scaling=1):
    if multi:
        losses_m = np.zeros((len(attrs), 4))
        for i in range(1,21):
            losses_m[i-1] = eval_multi_n(i, scaling=scaling)
        np.save(f"outputs/evaluation/n_multi{suffix}.npy",losses_m)

    if single:
        losses_s = np.zeros((len(attrs), 4))
        for i in range(1,21):
            losses_s[i-1] = eval_single_n(i)
        np.save(f"outputs/evaluation/n_single.npy",losses_s)


def plot_n_comparison(suffix="", suffixes=None):
    single = np.load("outputs/evaluation/n_single.npy").T

    loss_titles = ["Total", "Class", "Attr_Reg", "Identity_Recon"]

    for i, title in enumerate(loss_titles):
        plt.figure(figsize=(12,8))

        if suffixes:
            for s in suffixes:
                multi = np.load(f"outputs/evaluation/n_multi{s}.npy").T
                plt.plot(range(1,len(multi[i])+1), multi[i], label=f"MultiAttr: {s}")
        else:
            multi = np.load(f"outputs/evaluation/n_multi{suffix}.npy").T
            plt.plot(range(1,len(multi[i])+1), multi[i], label="MultiAttr")

        plt.plot(range(1,len(single[i])+1), single[i], label="SingleAttr")
        plt.title(title, fontsize=16)
        plt.xticks(range(1,len(single[i])+1))
        plt.legend(fontsize=12)
        plt.xlabel("Number of attributes", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.savefig(f"out_images/{exp}/n_eval_{title}.png", bbox_inches="tight")
        plt.clf()


def plot_overall(suffix=""):
    single = eval_single(save_img=False)
    multi = eval_multi(save_img=False)

    # Mock values
    # multi = [1,2,3,4]
    # single = [1,2,3,4]

    fig, ax = plt.subplots(figsize=(10,6))
    bar_width = 0.35
    opacity = 0.8
    loss_titles = ["Total", "Class", "Attr_Reg", "Identity_Recon"]

    x = np.arange(4)

    rects1 = ax.bar(x - bar_width/2, multi, bar_width,
                alpha=opacity,
                color='#FF5733',
                label='Multi Attribute')
    rects2 = ax.bar(x + bar_width/2, single, bar_width,
                    alpha=opacity,
                    color='#581845',
                    label='Single Attribute')

    ax.set_xlabel('Loss type', fontsize=12)
    ax.set_ylabel('Loss value', fontsize=12)
    ax.set_title('Loss comparison Multi vs Single', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(loss_titles)
    ax.set_yscale('log')
    ax.legend()

    plt.savefig(f"out_images/{exp}/eval_overall{suffix}.png", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    plot_overall(suffix="_vs_original")

    scaling_exp = True
    if scaling_exp:
        scaling_factors = [1, 1.5, 2, 2.5, 3]
        for scaling in scaling_factors:
            all_n_experiment(multi=True, single=False, suffix=f"_{scaling}", scaling=scaling)

        suffixes = [f"_{s}" for s in scaling_factors]
    else:
        all_n_experiment(multi=True, single=True, suffix=f"_vs_original")
        suffixes = None
    
    plot_n_comparison(suffix="_vs_original", suffixes=suffixes)

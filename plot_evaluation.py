import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import textwrap

from PIL import Image
from constants import NUM_TO_ATTR
from utils.functions import clip_img

matplotlib.use("tkagg")

PAD = 0.05

def plot_ratios(ratios, labels, scales, output_dir="outputs/evaluation/",
                title="Comparisson of target change ratio", filename="ratio_comparison.png"):

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8,5))

    for ratio, label in zip(ratios, labels):
        plt.plot(ratio, scales, label=label, marker='.')

    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.xlabel("Target Change Ratio", fontsize=12)
    plt.ylabel("Scaling Factor", fontsize=12)
    plt.xlim(0 - PAD, 1 + PAD)
    plt.ylim(scales[0] - PAD, scales[-1] + PAD)
    plt.savefig(output_dir + filename)


def plot_recon_vs_reg(recons, regs, ratios, labels, scales, output_dir="outputs/evaluation/",
                title="Comparisson of target change ratio", filename="ratio_comparison.png"):

    os.makedirs(output_dir, exist_ok=True)

    # Concatenate all values for getting easier the min, max values
    recons_cat = np.concatenate(recons)
    regs_cat = np.concatenate(regs)

    plt.figure(figsize=(7,14))

    plt.subplot(2,1,1)
    for ratio, recon, label in zip(ratios, recons, labels):
        plt.plot(ratio, recon, label=label, marker='.')

    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.xlabel("Target Change Ratio", fontsize=12)
    plt.ylabel("Identity Preservation", fontsize=12)
    plt.xlim(0 - PAD, 1 + PAD)
    plt.ylim(recons_cat.min(), recons_cat.max())

    plt.subplot(2,1,2)
    for ratio, reg, label in zip(ratios, regs, labels):
        plt.plot(ratio, reg, label=label, marker='.')

    plt.legend(fontsize=12)
    plt.xlabel("Target Change Ratio", fontsize=12)
    plt.ylabel("Attribute Preservation", fontsize=12)
    plt.xlim(0 - PAD, 1 + PAD)
    plt.ylim(regs_cat.min(), regs_cat.max())
    # plt.ylim(0.8, 1 + PAD*0.1)

    plt.savefig(output_dir + filename)


def plot_nattr_evolution(recons, regs, ratios, labels, output_dir="outputs/evaluation/",
                         title="Comparisson of target change ratio", filename="ratio_comparison.png"):

    os.makedirs(output_dir, exist_ok=True)

    # Concatenate all values for getting easier the min, max values
    recons_cat = np.concatenate(recons)
    regs_cat = np.concatenate(regs)

    plt.figure(figsize=(7,14))

    plt.subplot(2,1,1)
    for ratio, recon, label in zip(ratios, recons, labels):
        plt.plot(recon, ratio, label=label, marker='.')

    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.ylabel("Target Change Ratio", fontsize=12)
    plt.xlabel("Identity Preservation", fontsize=12)
    plt.ylim(0 - PAD, 1 + PAD)
    plt.xlim(recons_cat.min(), recons_cat.max())

    plt.subplot(2,1,2)
    for ratio, reg, label in zip(ratios, regs, labels):
        plt.plot(reg, ratio, label=label, marker='.')

    plt.legend(fontsize=12)
    plt.ylabel("Target Change Ratio", fontsize=12)
    plt.xlabel("Attribute Preservation", fontsize=12)
    plt.ylim(0 - PAD, 1 + PAD)
    plt.xlim(regs_cat.max(), regs_cat.min())
    # plt.ylim(0.8, 1 + PAD*0.1)

    plt.savefig(output_dir + filename)


def plot_images(images, coeff, attrs, save_dir, titles):
    description = ""
    for c in coeff.nonzero():
        description += attrs[c.item()] + ("+ , " if coeff[c.item()] > 0 else "- , ")

    description = textwrap.wrap(description[:-3], width=60)
    description = "\n".join(description)

    fig, axs = plt.subplots(nrows=1, ncols=len(images), figsize=(10, 6))
    for i, (image, title) in enumerate(zip(images, titles)):
        img_tensor = clip_img(image)[0]
        ndarr = img_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        axs[i].imshow(im)
        axs[i].set_title(title)
        axs[i].axis('off')

    plt.subplots_adjust(wspace=0.05)
    fig.text(0.5, 0.15, description, ha='center', va='center', fontsize=15)
    plt.savefig(save_dir)
    plt.close()
    fig.clf()
    plt.clf()


def plot_images_table(ratios, coeff, attrs, save_dir):
    description = ""
    for c in coeff.nonzero():
        description += attrs[c.item()] + ("+ , " if coeff[c.item()] > 0 else "- , ")

    description = textwrap.wrap(description[:-3], width=60)
    description = "\n".join(description)

    colors = np.array([["#c9cba3" if val else "#f28482" for val in row] for row in ratios])

    columns = [attrs[c.item()] for c in coeff.nonzero()]
    rows = ["Multi", "Single"]

    fig = plt.figure()
    fig = plt.figure(figsize=(4 + len(columns)*2, 2 + 3 / 2.5))
    table = plt.table(cellText=ratios, colLabels=columns, rowLabels=rows, loc='center',
                      cellLoc='center', cellColours=colors)
    table.auto_set_font_size(False)
    table.scale(1, 2)
    table.set_fontsize(16)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.savefig(save_dir, dpi=200, bbox_inches='tight')
    plt.close()
    fig.clf()
    plt.clf()


def plot_performance(times, memories, files, labels, output_dir="outputs/performance/"):

    os.makedirs(output_dir, exist_ok=True)

    n_attrs = list(range(0,21,5))
    n_attrs[0] = 1

    # time plot
    plt.figure(figsize=(8,5))
    for time, label in zip(times, labels):
        plt.plot(time, n_attrs, label=label, marker='.')
    plt.title("Comparison of training time needed per iteration", fontsize=16)
    plt.legend(fontsize=12)
    plt.xlabel("Number of attributes learned", fontsize=12)
    plt.ylabel("Mean time per iteration (s)", fontsize=12)
    plt.savefig(output_dir + "time.png")
    plt.clf()

    # memory plot
    plt.figure(figsize=(8,5))
    for mem, label in zip(memories, labels):
        plt.plot(mem, n_attrs, label=label, marker='.')
    plt.title("Comparison of peak GPU memory usage in training", fontsize=16)
    plt.legend(fontsize=12)
    plt.xlabel("Number of attributes learned", fontsize=12)
    plt.ylabel("Peak GPU memory usage (MB)", fontsize=12)
    plt.savefig(output_dir + "memory.png")
    plt.clf()

    # file plot
    plt.figure(figsize=(8,5))
    for file, label in zip(files, labels):
        plt.plot(file, n_attrs, label=label, marker='.')
    plt.title("Comparison of total model file output size", fontsize=16)
    plt.legend(fontsize=12)
    plt.xlabel("Number of attributes learned", fontsize=12)
    plt.ylabel("Total model size (MB)", fontsize=12)
    plt.savefig(output_dir + "file.png")
    plt.clf()

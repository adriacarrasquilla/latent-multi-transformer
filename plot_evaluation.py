import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.use("tkagg")

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

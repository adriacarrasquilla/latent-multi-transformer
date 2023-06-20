import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import textwrap

from PIL import Image
from utils.functions import clip_img

matplotlib.use("tkagg")

PAD = 0.05

def plot_ratios(ratios, labels, scales, output_dir="outputs/evaluation/",
                title="Comparisson of target change ratio", filename="ratio_comparison.png", figsize=(8,5)):

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=figsize)

    for ratio, label in zip(ratios, labels):
        plt.plot(ratio, scales, label=label, marker='.')

    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.xlabel("Target Change Ratio", fontsize=12)
    plt.ylabel("Scaling Factor", fontsize=12)
    plt.xlim(0 - PAD, 1 + PAD)
    plt.ylim(scales[0] - PAD, scales[-1] + PAD)
    plt.tight_layout()
    plt.savefig(output_dir + filename)


def plot_ratios_row(all_ratios, labels, scales, titles, output_dir="outputs/evaluation/",
                    filename="ratio_comparison.png"):

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, len(all_ratios), figsize=(4*len(all_ratios), 4), sharex=True, sharey=True)

    for i, (ratios, title) in enumerate(zip(all_ratios, titles)):
        for ratio, label in zip(ratios, labels):
            ax[i].plot(ratio, scales, label=label, marker='.')

        ax[i].set_title(title, fontsize=16)
        ax[i].set_xlim(0 - PAD, 1 + PAD)
        ax[i].set_ylim(scales[0] - PAD, scales[-1] + PAD)
        if i == 0:
            ax[i].set_ylabel("Scaling Factor", fontsize=12)
            ax[i].set_xlabel(" ", fontsize=12)
        if i == len(all_ratios) - 1:
            ax[i].legend(fontsize=12)

    fig.text(0.5, 0.02, 'Target Change Ratio', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir + filename)


def plot_recon_vs_reg_row(all_recons, all_regs, all_ratios, labels, titles,
                          output_dir="outputs/evaluation/", filename="ratio_comparison.png"):

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(5,10))
    fig, ax = plt.subplots(2, len(all_recons), figsize=(4*len(all_recons), 8), sharex=True)

    regs_cat = np.concatenate(all_regs)

    for i, (recons, regs, ratios, title) in enumerate(zip(all_recons, all_regs, all_ratios, titles)):
        # Concatenate all values for getting easier the min, max values
        recons_cat = np.concatenate(recons)

        # Identity Preservation Row
        for ratio, recon, label in zip(ratios, recons, labels):
            ax[0,i].plot(ratio, recon, label=label, marker='.')

        ax[0,i].set_title(title, fontsize=16)
        ax[0,i].set_xlim(0 - PAD, 1 + PAD)
        ax[0,i].set_ylim(recons_cat.min(), recons_cat.max())

        if i == len(all_recons) - 1:
            ax[0,i].legend(fontsize=12)

        if i == 0:
            ax[0,i].set_ylabel("Identity Preservation (IP)", fontsize=12)

        # Attribute Preservation Row
        for ratio, reg, label in zip(ratios, regs, labels):
            ax[1,i].plot(ratio, reg, label=label, marker='.')

        ax[1,i].set_xlim(0 - PAD, 1 + PAD)
        ax[1,i].set_ylim(regs_cat.min(), regs_cat.max())
        ax[1,i].set_xlabel(" ", fontsize=12)

        if i == 0:
            ax[1,i].set_ylabel("Attribute Preservation (AP)", fontsize=12)

        if i == len(all_recons) - 1:
            ax[1,i].legend(fontsize=12)


    fig.text(0.5, 0.02, 'Target Change Ratio', ha='center', fontsize=16)
    fig.subplots_adjust(hspace=0.1, wspace=0.2, bottom=0.2)
    plt.tight_layout()
    plt.savefig(output_dir + filename)


def plot_ratios_stacked(single_ratios, multi_ratios, attrs, scales, output_dir="outputs/evaluation/"):

    # Create a figure with subplots for each group of 4 elements
    num_plots = len(single_ratios) // 5
    print(num_plots)
    fig, axs = plt.subplots(num_plots, 5, sharey=True, figsize=(15, 10))

    multis, singles = [], []

    # Iterate over the data and create subplots
    for i in range(num_plots):
        for j in range(5):
            ax = axs[i, j] if num_plots > 1 else axs[j]
            sing, = ax.plot(single_ratios[i * 5 + j], scales, label="Single", marker='.')
            mult, = ax.plot(multi_ratios[i * 5 + j], scales, label="Multi", marker='.')
            ax.set_title(attrs[i * 5 + j])
            ax.set_xlim(0 - PAD, 1 + PAD)
            ax.set_ylim(scales[0] - PAD, scales[-1] + PAD)
            multis.append(mult)
            singles.append(sing)
            if i == num_plots - 1 and j == 4:
                ax.legend()

    # Set common y-axis label
    fig.text(0.08, 0.5, 'Scaling Factor', va='center', rotation='vertical', fontsize=12)

    # Set common x-axis label
    fig.text(0.5, 0.06, 'Target Change Ratio', ha='center', fontsize=12)

    # legend_handles = [
    #     mpatches.Patch(color='blue', label='Single'),
    #     mpatches.Patch(color='orange', label='Multi')
    # ]
    #
    # # Create a general legend
    # fig.legend(handles=legend_handles)

    # Adjust the spacing between subplots
    fig.subplots_adjust(wspace=0.2, hspace=0.4)

    # Display the figure
    plt.savefig(output_dir + f"individual_all.png")


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

def plot_corr_vs_uncorr(ratios, corr, uncorr, labels, scales, output_dir="outputs/evaluation/",
                title="Comparisson of target change ratio", filename="ratio_comparison.png"):

    os.makedirs(output_dir, exist_ok=True)

    # Concatenate all values for getting easier the min, max values
    corr_cat = np.concatenate(corr)
    uncorr_cat = np.concatenate(uncorr)

    plt.figure(figsize=(7,14))

    plt.subplot(2,1,1)
    for ratio, recon, label in zip(ratios, corr, labels):
        plt.plot(ratio, recon, label=label, marker='.')

    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.xlabel("Target Change Ratio", fontsize=12)
    plt.ylabel("Correlated attr preservation", fontsize=12)
    plt.xlim(0 - PAD, 1 + PAD)
    plt.ylim(corr_cat.min(), corr_cat.max())

    plt.subplot(2,1,2)
    for ratio, reg, label in zip(ratios, uncorr, labels):
        plt.plot(ratio, reg, label=label, marker='.')

    plt.legend(fontsize=12)
    plt.xlabel("Target Change Ratio", fontsize=12)
    plt.ylabel("Uncorrelated attr preservation", fontsize=12)
    plt.xlim(0 - PAD, 1 + PAD)
    plt.ylim(uncorr_cat.min(), uncorr_cat.max())
    # plt.ylim(0.8, 1 + PAD*0.1)

    plt.savefig(output_dir + filename)


def plot_nattr_evolution_fixed_attr(recons, regs, ratios, labels, n_attrs,
                                     output_dir="outputs/evaluation/", filename="ratio_comparison.png"):

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(5,10))
    fig, ax = plt.subplots(2, len(n_attrs), figsize=(4*len(n_attrs), 8), sharex=True)

    for i, n in enumerate(n_attrs):
        # Concatenate all values for getting easier the min, max values
        recons_cat = np.concatenate(recons[n,:,:])
        regs_cat = np.concatenate(regs[n_attrs,:,:])

        # Identity Preservation Row
        for ratio, recon, label in zip([ratios[n, 0, :], ratios[n, 1, :]],
                                       [recons[n, 0, :], recons[n, 1, :]],
                                       labels):
            ax[0,i].plot(ratio, recon, label=label, marker='.')

        ax[0,i].set_title(f"{n+1} attributes", fontsize=16)
        ax[0,i].set_xlim(0 - PAD, 1 + PAD)
        ax[0,i].set_ylim(recons_cat.min(), recons_cat.max())

        if i == len(n_attrs) - 1:
            ax[0,i].legend(fontsize=12)

        if i == 0:
            ax[0,i].set_ylabel("Identity Preservation (IP)", fontsize=12)

        # Attribute Preservation Row
        for ratio, reg, label in zip([ratios[n,0, :], ratios[n, 1, :]],
                                     [regs[n, 0, :], regs[n, 1, :]],
                                     labels):
            ax[1,i].plot(ratio, reg, label=label, marker='.')

        ax[1,i].set_xlim(0 - PAD, 1 + PAD)
        ax[1,i].set_ylim(regs_cat.min(), regs_cat.max())
        ax[1,i].set_xlabel(" ", fontsize=12)

        if i == 0:
            ax[1,i].set_ylabel("Attribute Preservation (AP)", fontsize=12)

        if i == len(n_attrs) - 1:
            ax[1,i].legend(fontsize=12)


    # fig.text(0.5, 0.5, 'Identity Preservation', ha='center', fontsize=14)
    fig.text(0.5, 0.02, 'Target Change Ratio', ha='center', fontsize=16)
    fig.subplots_adjust(hspace=0.1, wspace=0.2, bottom=0.2)
    plt.tight_layout()
    plt.savefig(output_dir + filename)

def plot_nattr_evolution_fixed_coeff(recons, regs, ratios, labels, n_attrs, coeffs,
                                     output_dir="outputs/evaluation/", filename="ratio_comparison.png"):

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(5,10))
    fig, ax = plt.subplots(2, len(n_attrs), figsize=(4*len(n_attrs), 8), sharey=False)

    for i, n in enumerate(n_attrs):
        # Concatenate all values for getting easier the min, max values
        recons_cat = np.concatenate(recons[:,:,n])
        regs_cat = np.concatenate(regs[:,:,n])

        for ratio, recon, label in zip([ratios[:,0, n], ratios[:, 1, n]],
                                       [recons[:, 0, n], recons[:, 1, n]],
                                       labels):
            ax[0,i].plot(recon, ratio, label=label, marker='.')


        if i == 0:
            ax[0,i].set_ylabel("Target Change Ratio", fontsize=12)

        ax[0,i].set_title(f"Scaling Factor: {coeffs[n]:.2}", fontsize=16)
        ax[0,i].set_ylim(0 - PAD, 1 + PAD)
        ax[0,i].set_xlim(recons_cat.min(), recons_cat.max())

        if i == len(n_attrs) - 1:
            ax[0,i].legend(fontsize=12)

        for ratio, reg, label in zip([ratios[:,0, n], ratios[:, 1, n]],
                                     [regs[:, 0, n], regs[:, 1, n]],
                                     labels):
            ax[1,i].plot(reg, ratio, label=label, marker='.')

        if i == 0:
            ax[1,i].set_ylabel("Target Change Ratio", fontsize=12)
        if i == len(n_attrs) - 1:
            ax[1,i].legend(fontsize=12)

        ax[1,i].set_ylim(0 - PAD, 1 + PAD)
        ax[1,i].set_xlim(regs_cat.max(), regs_cat.min())
        ax[1,i].set_xlabel(" ", fontsize=12)

    fig.text(0.5, 0.5, 'Identity Preservation', ha='center', fontsize=14)
    fig.text(0.5, 0.02, 'Attribute Preservation', ha='center', fontsize=14)
    fig.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.2)
    plt.tight_layout(h_pad=4)
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

    n_attrs = list(range(0,21,4))
    n_attrs[0] = 1

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # time plot
    # plt.figure(figsize=(5,5))
    for time, label in zip(times, labels):
        ax[0].plot(n_attrs, time, label=label, marker='.')
    ax[0].set_title("Training time needed per iteration", fontsize=12)
    ax[0].legend(fontsize=12)
    ax[0].set_xlabel("Number of attributes learned", fontsize=10)
    ax[0].set_ylabel("Mean time per iteration (s)", fontsize=10)
    ax[0].set_xticks(n_attrs)
    # plt.savefig(output_dir + "time.png")
    # plt.clf()

    # memory plot
    # plt.figure(figsize=(5,5))
    for mem, label in zip(memories, labels):
        ax[1].plot(n_attrs, mem, label=label, marker='.')
    ax[1].set_title("Peak GPU memory usage", fontsize=12)
    ax[1].legend(fontsize=12)
    ax[1].set_xlabel("Number of attributes learned", fontsize=10)
    ax[1].set_ylabel("Peak GPU memory usage (MB)", fontsize=10)
    ax[1].set_xticks(n_attrs)
    # plt.savefig(output_dir + "memory.png")
    # plt.clf()

    # file plot
    # plt.figure(figsize=(5,5))
    for file, label in zip(files, labels):
        ax[2].plot(n_attrs, file, label=label, marker='.')
    ax[2].set_title("Total model file output size", fontsize=12)
    ax[2].legend(fontsize=12)
    ax[2].set_xlabel("Number of attributes learned", fontsize=10)
    ax[2].set_ylabel("Total model size (MB)", fontsize=10)
    ax[2].set_xticks(n_attrs)
    # plt.savefig(output_dir + "file.png")
    plt.tight_layout(pad=2.5)
    plt.savefig(output_dir + "performance.png")
    # plt.clf()


def plot_random_images(images, coeff, attrs, save_dir):

    fig, axs = plt.subplots(nrows=2, ncols=len(images), figsize=(5*len(images), 10))

    for i, image_pair in enumerate(images):
        for j, image in enumerate(image_pair):
            img_tensor = clip_img(image)[0]
            ndarr = img_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            axs[j][i].imshow(im)
            axs[j][i].axis('off')

        if coeff[i] == 0:
            notation = ""
        else:
            notation = "+" if coeff[i] > 0 else "-"
        axs[0][i].set_title(attrs[i] + notation, fontsize=20)


    fig.text(0.11, 0.67, 'Multi', va='center', fontsize=20, rotation="vertical")
    fig.text(0.11, 0.3, 'Single', va='center', fontsize=20, rotation="vertical")
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig(save_dir)
    plt.close()
    fig.clf()
    plt.clf()

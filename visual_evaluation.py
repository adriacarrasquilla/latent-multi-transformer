import argparse
import os
import gc
import numpy as np
import torch
import yaml
from rich.progress import track

from datasets import *
from utils.functions import *
from plot_evaluation import plot_images, plot_images_table
from evaluation import apply_transformation, get_ratios_from_sample, get_trainer, evaluate_scaling_vs_change_ratio

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
n_steps = 2
scale = 1.0
n_samples = 500

log_dir_single = os.path.join(opts.log_path, "original_train") + "/"

save_dir = os.path.join("./outputs/evaluation/", opts.out) + "/"
os.makedirs(save_dir, exist_ok=True)

log_dir = os.path.join(opts.log_path, opts.config) + "/"
config = yaml.safe_load(open("./configs/" + opts.config + ".yaml", "r"))

attrs = config["attr"].split(",")
attr_num = [ATTR_TO_NUM[a] for a in attrs]


def get_individual_scores(coeff_i):
    single_rates, single_recons, single_regs = evaluate_scaling_vs_change_ratio(multi=False, return_mean=False, n_samples=n_samples,
                                                                                n_steps=n_steps, scale=scale)
    multi_rates, multi_recons, multi_regs = evaluate_scaling_vs_change_ratio(multi=True, return_mean=False, n_samples=n_samples,
                                                                             n_steps=n_steps, scale=scale)
    s_rates = single_rates[:, coeff_i]
    m_rates = multi_rates[:, coeff_i]
    s_recons = single_recons[:, coeff_i]
    m_recons = multi_recons[:, coeff_i]
    s_regs = single_regs[:, coeff_i]
    m_regs = multi_regs[:, coeff_i]

    return [s_rates, m_rates], [s_recons, m_recons], [s_regs, m_regs]

def get_diff_examples(scores, diff_type="ratio"):
    # Output directory
    examples_dir = save_dir + "examples/"
    os.makedirs(examples_dir, exist_ok=True)

    # Evaluate models
    all_coeffs = np.load(testdata_dir + "labels/overall.npy")

    # Use only one scaling factor and separate multi from single results
    scales = torch.linspace(0, scale, n_steps)
    coeff_i = 1

    s_rates, m_rates = scores[0][0], scores[0][1]
    s_recons, m_recons = scores[1][0], scores[1][1]
    s_regs, m_regs = scores[2][0], scores[2][1]

    # Get diff and n best results for both multi and single
    if diff_type == "ratio":
        diff = s_rates-m_rates
    elif diff_type == "identity":
        diff = s_recons-m_recons
    else: # attr_p
        diff = s_regs-m_regs

    n_best = 2
    indices = np.where(np.sum(np.abs(all_coeffs[:n_samples]), axis=1) > 3)[0]
    best_multi = indices[np.argpartition(diff[indices], n_best)[:n_best]]
    best_single = indices[np.argpartition(-diff[indices], n_best)[:n_best]]

    # Generate images and prepare plot comparison for each result
    for k in track(np.concatenate((best_multi, best_single)), f"Generating images for {diff_type}"):
        w_0 = np.load(testdata_dir + "latent_code_%05d.npy" % k)
        w_0 = torch.tensor(w_0).to(DEVICE)

        coeff = torch.tensor(all_coeffs[k]).to(DEVICE)
        coeff = coeff * scales[coeff_i]

        # Generate the original image first
        trainer = get_trainer(multi=True)
        x_0, _ = trainer.StyleGAN([w_0], input_is_latent=True, randomize_noise=False)
        x_1 = x_0
        img_l = [x_0]
        titles = ["Original"]

        ratios = []

        # Generate multi and single transformations
        for multi in [True, False]:
            trainer = get_trainer(multi=multi)
            w_1 = apply_transformation(trainer=trainer, w_0=w_0, coeff=coeff, multi=multi)

            # Get ratio before tweaking w_1 for image generation
            ratio = get_ratios_from_sample(w_0, w_1, coeff, trainer)
            ratios.append(ratio)

            # Convert latent space to image
            w_1 = torch.cat((w_1[:,:11,:], w_0[:,11:,:]), 1)
            x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)
            img_l.append(x_1)

            # Write info about performance for each result in the titles
            if multi:
                title = f"Target Change (%): {m_rates[k]:.2} \n" \
                        f"Identity (MSE): {m_recons[k]:.2} \n" \
                        f"Attr Unchanged (%): {m_regs[k]:.2} \n Multi"
            else:
                title = f"Target Change (%): {s_rates[k]:.2} \n" \
                        f"Identity (MSE): {s_recons[k]:.2} \n" \
                        f"Attr Unchanged (%): {s_regs[k]:.2}\n Single"

            titles.append(title)



        filename = f"diff_{diff_type}_{k}.jpg"
        filename_table = f"table_{diff_type}_{k}.jpg"

        plot_images(img_l, save_dir=examples_dir + filename, coeff=coeff.squeeze(0), titles=titles, attrs=attrs)

        # Free up GPU memory
        del img_l, x_0, x_1
        torch.cuda.empty_cache()
        gc.collect()

        ratios = torch.stack(ratios).cpu().tolist()
        plot_images_table(ratios=ratios, coeff=coeff.squeeze(0), attrs=attrs, save_dir=examples_dir + filename_table)


def get_all_diff_types_examples():
    scores = get_individual_scores(coeff_i=1)
    for diff_type in ["ratio", "identity", "attr_p"]:
        get_diff_examples(scores, diff_type=diff_type)


if __name__ == "__main__":
    get_all_diff_types_examples()

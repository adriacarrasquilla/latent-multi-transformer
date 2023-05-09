import random
from rich.progress import track
from torchvision import os
import yaml

import numpy as np
import torch
from constants import ATTR_TO_NUM, NUM_TO_ATTR, ATTR_TO_SPANISH
from evaluation import apply_transformation, get_trainer
from torchvision import utils

from nets import LCNet
from trainer import Trainer as MultiTrainer
from original_trainer import Trainer as SingleTrainer
from utils.functions import clip_img

random.seed(1)
testdata_dir = "./data/ffhq/"
out_dir = testdata_dir + "labels/"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

classifier_model_path = "./models/latent_classifier_epoch_20.pth"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
DEVICE = torch.device("cuda")

classifier = LCNet([9216, 2048, 512, 40], activ="leakyrelu")
classifier.load_state_dict(torch.load(classifier_model_path, map_location=DEVICE))
classifier.eval()
classifier.to(DEVICE)

conf_file = '15_attrs'
config = yaml.safe_load(open('./configs/' + conf_file + '.yaml', 'r'))
all_attrs = [ATTR_TO_NUM[a] for a in config["attr"].split(',')]

n_samples = 1000

def overall_dataset(n_samples=n_samples, all_attrs=all_attrs, name="overall"):
    all_coeffs = np.zeros((n_samples,len(all_attrs)), dtype=np.int8)

    for k in range(n_samples):
        local_attrs = random.sample(range(len(all_attrs)), random.randint(1,len(all_attrs)))
        global_attrs = [all_attrs[a] for a in local_attrs]
        w_0 = np.load(testdata_dir + "latent_code_%05d.npy" % k)
        w_0 = torch.tensor(w_0).to(DEVICE)

        predict_lbl_0 = classifier(w_0.view(w_0.size(0), -1))
        lbl_0 = torch.sigmoid(predict_lbl_0)
        attr_pb_0 = lbl_0[torch.arange(lbl_0.shape[0]), global_attrs]
        coeff = torch.where(attr_pb_0 > 0.5, -1, 1).detach().cpu().numpy()
        all_coeffs[k][local_attrs] = coeff

    np.save(out_dir + f"{name}.npy", all_coeffs)


def individual_dataset(n_samples=n_samples, all_attrs=all_attrs):
    all_coeffs = np.zeros((n_samples,len(all_attrs)), dtype=np.int8)

    for k in range(n_samples):
        w_0 = np.load(testdata_dir + "latent_code_%05d.npy" % k)
        w_0 = torch.tensor(w_0).to(DEVICE)

        predict_lbl_0 = classifier(w_0.view(w_0.size(0), -1))
        lbl_0 = torch.sigmoid(predict_lbl_0)
        attr_pb_0 = lbl_0[torch.arange(lbl_0.shape[0]), all_attrs]

        coeff = torch.where(attr_pb_0 > 0.5, -1, 1).detach().cpu().numpy()
        all_coeffs[k] = coeff

    np.save(out_dir + "all.npy", all_coeffs)


def attributes_order_dataset(n_samples=n_samples, all_attrs=all_attrs):
    all_orders = np.zeros((n_samples,len(all_attrs)), dtype=np.int8)
    for k in range(n_samples):
        all_orders[k] = np.array(random.sample(range(len(all_attrs)), len(all_attrs)))

    np.save(out_dir + "attr_order.npy", all_orders)


def subjective_study(n_imgs=20, all_attrs=all_attrs):

    random.seed(666)
    out = "./data/subjective/"
    os.makedirs(out + "img/", exist_ok=True)
    os.makedirs(out + "txt/", exist_ok=True)

    all_coeffs = np.load(testdata_dir + "labels/all.npy")
    narrow_done = False
    for i, k in track(enumerate(random.sample(range(n_samples), n_imgs)), "Generating Samples"):

        attr_idx = list(range(len(all_attrs)))

        if narrow_done:
            # Narrow eyes are too present in the results. Engineering this to remove it (they both do equally good)
            attr_idx.remove(16)

        local_attrs = random.sample(attr_idx, random.randint(1, len(all_attrs)))

        if 16 in local_attrs:
            narrow_done = True

        coeff = torch.zeros(all_coeffs[k].shape).to(DEVICE)
        coeff[local_attrs] = torch.tensor(all_coeffs[k][local_attrs], dtype=torch.float).to(DEVICE)

        w_0 = np.load(testdata_dir + "latent_code_%05d.npy" % k)
        w_0 = torch.tensor(w_0).to(DEVICE)

        # multi + original
        trainer = get_trainer(multi=True)

        x_0, _ = trainer.StyleGAN([w_0], input_is_latent=True, randomize_noise=False)
        utils.save_image(clip_img(x_0), out + f"img/{i+1}_original.jpg")

        w_1 = apply_transformation(trainer, w_0, coeff, multi=True)
        w_1 = torch.cat((w_1[:,:11,:], w_0[:,11:,:]), 1)
        x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)
        utils.save_image(clip_img(x_1), out + f"img/{i+1}_multi.jpg")

        # single
        trainer = get_trainer(multi=False)
        w_1 = apply_transformation(trainer, w_0, coeff, multi=False)
        w_1 = torch.cat((w_1[:,:11,:], w_0[:,11:,:]), 1)
        x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)
        utils.save_image(clip_img(x_1), out + f"img/{i+1}_single.jpg")

        with open(out + f"txt/{i+1}_attrs.txt", "w") as f:
            for attr in local_attrs:
                f.write(f"{NUM_TO_ATTR[all_attrs[attr]]}: {coeff[attr]}\n")

def subjective_form_assets(n_imgs=20, AB=False):
    out = "./data/subjective/form/"
    os.makedirs(out, exist_ok=True)

    if AB:
        with open(out + "transformer_order.csv", "w") as f:
            f.write("question,multi,single\n")
            for i in range(n_imgs):
                multi, single = random.sample(["A", "B"], 2)
                f.write(f"{i+1},{multi},{single}\n")

    for i in range(1,21):
        print(f"Question {i}")
        operations = {"add": [], "remove": []}
        operations_spanish = {"Añadimos": [], "Eliminamos": []}
        with open("./data/subjective/best/txt/" + f"{i}_attrs.txt", "r") as f:
            for line in f.readlines():
                attr, coeff = line[:-1].split(":")
                op = 'remove' if "-" in coeff else 'add'
                op_s = 'Eliminamos' if "-" in coeff else 'Añadimos'
                operations[op].append(attr)
                operations_spanish[op_s].append(ATTR_TO_SPANISH[attr])

        for key in operations_spanish.keys():
            print(f"{key} los siguientes atributos: {', '.join(operations_spanish[key])}")

        print()
        for key in operations.keys():
            print(f"We {key} the following attributes: {', '.join(operations[key])}")

        print("-------------------------------------------------------------------")


# individual_dataset()
overall_dataset(name=conf_file)
# attributes_order_dataset()
# subjective_study()
# subjective_form_assets()

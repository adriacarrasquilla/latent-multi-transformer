import random
from torchvision import os
import yaml

import numpy as np
import torch
from constants import ATTR_TO_NUM

from nets import LCNet

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

conf_file = 'new_train'
config = yaml.safe_load(open('./configs/' + conf_file + '.yaml', 'r'))
all_attrs = [ATTR_TO_NUM[a] for a in config["attr"].split(',')]

n_samples = 1000

def overall_dataset(n_samples=n_samples, all_attrs=all_attrs):
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

    np.save(out_dir + "overall.npy", all_coeffs)


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


# individual_dataset()
# overall_dataset()
attributes_order_dataset()

# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import argparse
import os
import numpy as np
import torch
import torch.utils.data as data
import yaml

from PIL import Image
from torch.utils.tensorboard.writer import SummaryWriter

# Set up to allow both gpu and cpu runtimes
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda')

from datasets import *
from original_trainer import *
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='scaling', help='Path to the config file.')
parser.add_argument('--latent_path', type=str, default='./data/celebahq_dlatents_psp.npy', help='dataset path')
parser.add_argument('--label_file', type=str, default='./data/celebahq_anno.npy', help='label file path')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='stylegan model path')
parser.add_argument('--classifier_model_path', type=str, default='./models/latent_classifier_epoch_20.pth', help='pretrained attribute classifier')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
parser.add_argument('--multigpu', type=bool, default=False, help='use multiple gpus')
parser.add_argument('--attr', type=str, default=None, help='Override config')
opts = parser.parse_args()

# Celeba attribute list
attr_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, \
            'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, \
            'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, \
            'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, \
            'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, \
            'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, \
            'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, \
            'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

log_dir = os.path.join(opts.log_path, opts.config) + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = SummaryWriter(log_dir=log_dir)

config = yaml.safe_load(open('./configs/' + opts.config + '.yaml', 'r'))
attr_l = config['attr'].split(',')

if  opts.attr:
    attr_l = [opts.attr]

batch_size = config['batch_size']
epochs = config['epochs']

@profile
def get_latents():
    w = np.load(opts.latent_path)
    return torch.tensor(w).to(DEVICE)

w = get_latents()

@profile
def get_loader():
    dataset_A = LatentDataset(opts.latent_path, opts.label_file, training_set=True)
    loader_A = data.DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    return loader_A

loader_A = get_loader()


@profile
def train_model_parallel(attr, loader_A=loader_A):
    print(f"Training attribute {attr}")

    total_iter = 0
    attr_num = attr_dict[attr]
    print(attr_num)

    # Initialize trainer
    trainer = Trainer(config, attr_num, attr, opts.label_file)
    trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)   
    trainer.to(DEVICE)

    for _ in range(epochs):

        for n_iter, list_A in enumerate(loader_A):
            w_A, lbl_A = list_A
            w_A, lbl_A = w_A.to(DEVICE), lbl_A.to(DEVICE)
            trainer.update(w_A, None, n_iter)

            total_iter += 1

            if n_iter == 300:
                break

        trainer.save_model(log_dir)

    return 1


if __name__ == "__main__":
    # num_processes = multiprocessing.cpu_count()
    # num_processes = 1
    # pool = multiprocessing.Pool(processes=num_processes)

    print('Start training!')
    # results = [pool.apply(train_model_parallel, args=(attr)) for attr in attr_l[:num_processes]]
    # results = pool.map(train_model_parallel, attr_l[:num_processes])
    #
    # pool.close()
    # pool.join()

    results = train_model_parallel(opts.attr)

    print(results)
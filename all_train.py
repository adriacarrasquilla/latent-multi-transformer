# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import time
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml

from PIL import Image
from torchvision import transforms, utils
from torch.utils.tensorboard.writer import SummaryWriter

# Set up to allow both gpu and cpu runtimes
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device('cuda')
else:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "Turing"
    DEVICE = torch.device('cpu')

from datasets import *
from trainer import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='new_train', help='Path to the config file.')
parser.add_argument('--latent_path', type=str, default='./data/celebahq_dlatents_psp.npy', help='dataset path')
parser.add_argument('--label_file', type=str, default='./data/celebahq_anno.npy', help='label file path')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='stylegan model path')
parser.add_argument('--classifier_model_path', type=str, default='./models/latent_classifier_epoch_20.pth', help='pretrained attribute classifier')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
parser.add_argument('--multigpu', type=bool, default=False, help='use multiple gpus')
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
attrs = config['attr'].split(',')
batch_size = config['batch_size']
epochs = config['epochs']

dlatents = np.load(opts.latent_path)
w = torch.tensor(dlatents).to(DEVICE)

dataset_A = LatentDataset(opts.latent_path, opts.label_file, training_set=True)
loader_A = data.DataLoader(dataset_A, batch_size=batch_size, shuffle=True)

print('Start training!')
times = []

print(f"Training model for: {','.join(attrs)}")

total_iter = 0
# attr_num = attr_dict[attr1]
attr_num = [attr_dict[a] for a in attrs]

# Initialize trainer
trainer = Trainer(config, attr_num, attrs, opts.label_file)
trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)   
trainer.to(DEVICE)

for n_epoch in range(epochs):
    t = time.time()

    for n_iter, list_A in enumerate(loader_A):
        w_A, lbl_A = list_A
        w_A, lbl_A = w_A.to(DEVICE), lbl_A.to(DEVICE)
        trainer.update(w_A, None, n_iter)

        if (total_iter+1) % config['log_iter'] == 0:
            trainer.log_loss(logger, total_iter)
        if (total_iter+1) % config['image_log_iter'] == 0:
            trainer.log_image(logger, w[total_iter%dataset_A.length].unsqueeze(0), total_iter)

        total_iter += 1

        # if n_iter == 10000:
        #     break

    trainer.save_model_multi(log_dir, name="20_attrs")
    e_time = time.time() - t
    times.append(e_time)
    print(f"Time for epoch {n_epoch}: {e_time}")

print(f"Mean time per epochs (over {epochs}): {np.mean(times)}")

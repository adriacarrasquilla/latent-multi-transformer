# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import argparse
import copy
import glob
import numpy as np
import os
import torch
import yaml
import time 

from PIL import Image
from torchvision import transforms, utils, models

from utils.video_utils import *
from trainer import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
DEVICE = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
parser.add_argument('--attr', type=str, default='Eyeglasses', help='attribute for manipulation.')
parser.add_argument('--alpha', type=str, default='1.', help='scale for manipulation.')
parser.add_argument('--label_file', type=str, default='./data/celebahq_anno.npy', help='label file path')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='stylegan model path')
parser.add_argument('--classifier_model_path', type=str, default='./models/latent_classifier_epoch_20.pth', help='pretrained attribute classifier')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--function', type=str, default='', help='Calling function by name.')
parser.add_argument('--video_path', type=str, default='./data/video/FP006911MD02.mp4', help='video file path')
parser.add_argument('--output_path', type=str, default='./outputs/video/', help='output video file path')
parser.add_argument('--optical_flow', action='store_true', help='use optical flow')
parser.add_argument('--resize', action='store_true', help='downscale image size')
parser.add_argument('--seamless', action='store_true', help='seamless cloning')
parser.add_argument('--filter_size', type=float, default=3, help='filter size')
parser.add_argument('--strs', type=str, default='Original,Projected,Manipulated', help='strs to be added on video')
opts = parser.parse_args()

# Celeba attribute list
attr_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, \
            'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, \
            'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, \
            'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, \
            'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, \
            'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, \
            'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, \
            'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

# Latent code manipulation
def latent_manipulation(opts, latent_dir_path, process_dir_path):

    attrs = opts.attr.split(',')
    alphas = opts.alpha.split(',')
    os.makedirs(process_dir_path, exist_ok=True)

    with torch.no_grad():

        log_dir = os.path.join(opts.log_path, opts.config) + '/'
        config = yaml.safe_load(open('./configs/' + opts.config + '.yaml', 'r'))
        
        # Initialize trainer
        trainer = Trainer(config, None, None, opts.label_file)
        trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)
        trainer.to(DEVICE)

        latent_num = len(glob.glob1(latent_dir_path,'*.npy'))

        T_nets = []
        for attr_idx, attr in enumerate(attrs):
            trainer.attr_num = attr_dict[attr]
            trainer.load_model(log_dir)
            T_nets.append(copy.deepcopy(trainer.T_net))

        for k in range(latent_num):
            w_0 = np.load(latent_dir_path + 'latent_code_%05d.npy'%k)
            w_0 = torch.tensor(w_0).to(DEVICE)
            w_1 = w_0.clone()
            for attr_idx, attr in enumerate(attrs):
                alpha = torch.tensor(float(alphas[attr_idx]))
                w_1 = T_nets[attr_idx](w_1.view(w_0.size(0), -1), alpha.unsqueeze(0).to(DEVICE))
            w_1 = w_1.view(w_0.size())
            w_1 = torch.cat((w_1[:,:11,:], w_0[:,11:,:]), 1)
            x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)
            utils.save_image(clip_img(x_1), process_dir_path + 'frame%04d'%k+'.jpg')


video_path = opts.video_path
video_name = video_path.split('/')[-1]
orig_dir_path = opts.output_path + video_name.split('.')[0] + '/' + video_name.split('.')[0] + '/'
align_dir_path = os.path.dirname(orig_dir_path) + '_crop_align/'
latent_dir_path = os.path.dirname(orig_dir_path) + '_crop_align_latent/'
process_dir_path = os.path.dirname(orig_dir_path) + '_crop_align_' + opts.attr.replace(',','_') + '/'
reproject_dir_path = os.path.dirname(orig_dir_path) + '_crop_align_' + opts.attr.replace(',','_') + '_reproject/'


print(opts.function)
start_time = time.perf_counter()

if opts.function == 'video_to_frames':
    video_to_frames(video_path, orig_dir_path, resize=opts.resize)
    create_video(orig_dir_path)
elif opts.function == 'align_frames':
    align_frames(orig_dir_path, align_dir_path, output_size=1024, optical_flow=opts.optical_flow, filter_size=opts.filter_size)
elif opts.function == 'latent_manipulation':
    latent_manipulation(opts, latent_dir_path, process_dir_path)
elif opts.function == 'reproject_origin':
    process_dir_path = os.path.dirname(orig_dir_path) + '_crop_align_latent/inference_results/'
    reproject_dir_path = os.path.dirname(orig_dir_path) + '_crop_align_origin_reproject/'
    video_reproject(orig_dir_path, process_dir_path, reproject_dir_path, align_dir_path, seamless=opts.seamless)
    create_video(reproject_dir_path)
elif opts.function == 'reproject_manipulate':
    video_reproject(orig_dir_path, process_dir_path, reproject_dir_path, align_dir_path, seamless=opts.seamless)
    create_video(reproject_dir_path)
elif opts.function == 'compare_frames':
    process_dir_paths = []
    process_dir_paths.append(os.path.dirname(orig_dir_path) + '_crop_align_origin_reproject/')
    process_dir_paths.append(os.path.dirname(orig_dir_path) + '_crop_align_' + opts.attr.split(',')[0] + '_reproject/')
    if len(opts.attr.split(','))>1:
        process_dir_paths.append(reproject_dir_path)
    save_dir = os.path.dirname(orig_dir_path) + '_crop_align_' + opts.attr.replace(',','_') + '_compare/'
    compare_frames(save_dir, orig_dir_path, process_dir_paths, strs=opts.strs, dim=1)
    create_video(save_dir, video_format='.avi', resize_ratio=1)

count_time = time.perf_counter() - start_time
print("Elapsed time: %0.4f seconds"%count_time)

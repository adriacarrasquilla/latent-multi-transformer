# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import random
random.seed(1)

from torchvision import utils

from nets import F_mapping_multi, LCNet
from utils.functions import *

import sys 
sys.path.append('pixel2style2pixel/')

from pixel2style2pixel.models.stylegan2.model import Generator
from pixel2style2pixel.models.psp import get_keys

from constants import DEVICE

class Trainer(nn.Module):
    def __init__(self, config, attr_num, attr, label_file, scaling=1):
        super(Trainer, self).__init__()
        # Load Hyperparameters
        self.accumulation_steps = 16
        self.config = config

        # For multi label
        self.attr_nums = attr_num
        self.attrs = attr

        # for single label
        self.attr_num = attr_num[0]
        self.attr = attr[0]

        # To control inference only
        self.scaling = scaling

        mapping_lrmul = self.config['mapping_lrmul']
        mapping_layers = self.config['mapping_layers']
        mapping_fmaps = self.config['mapping_fmaps']
        mapping_nonlinearity = self.config['mapping_nonlinearity']
        # Networks
        # Latent Transformer
        self.T_net = F_mapping_multi(
            mapping_lrmul= mapping_lrmul,
            mapping_layers=mapping_layers,
            mapping_fmaps=mapping_fmaps,
            mapping_nonlinearity = mapping_nonlinearity,
            n_attributes=len(self.attrs)
        )
        # Latent Classifier
        self.Latent_Classifier = LCNet([9216, 2048, 512, 40], activ='leakyrelu')
        # StyleGAN Model
        self.StyleGAN = Generator(1024, 512, 8)

        self.label_file = label_file
        self.corr_ma = None

        # Optimizers
        self.params = list(self.T_net.parameters())

        self.optimizer = torch.optim.Adam(self.params, lr=config['lr'],
                                          betas=(config['beta_1'], config['beta_2']),
                                          weight_decay=config['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['step_size'],
                                                         gamma=config['gamma'])


    def initialize(self, stylegan_model_path, classifier_model_path):
        state_dict = torch.load(stylegan_model_path, map_location='cpu')
        self.StyleGAN.load_state_dict(get_keys(state_dict, 'decoder'), strict=True)
        self.Latent_Classifier.load_state_dict(torch.load(classifier_model_path, map_location=DEVICE))
        self.Latent_Classifier.eval()

    def L1loss(self, input, target):
        return nn.L1Loss()(input,target)
    
    def MSEloss(self, input, target):
        if isinstance(input, list):
            return sum([nn.MSELoss()(input[i],target[i]) for i in range(len(input))])/len(input)
        else:
            return nn.MSELoss()(input,target)

    def SmoothL1loss(self, input, target):
        return nn.SmoothL1Loss()(input, target)

    def CEloss(self, x, target, reduction='mean'):
        return nn.CrossEntropyLoss(reduction=reduction)(x, target)
    
    def BCEloss(self, x, target, reduction='mean'):
        return nn.BCEWithLogitsLoss(reduction=reduction)(x, target)

    def MultiLabelLoss(self, x, target, reduction='mean'):
        return nn.MultiLabelSoftMarginLoss(reduction=reduction)(x, target)

    def GAN_loss(self, x, real=True):
        if real:
            target = torch.ones(x.size()).type_as(x)
        else:
            target = torch.zeros(x.size()).type_as(x)
        return nn.MSELoss()(x, target)

    def get_correlation(self, attr_num, threshold=1):
        if self.corr_ma is None:
            lbls = np.load(self.label_file)
            self.corr_ma = np.corrcoef(lbls.transpose())
            self.corr_ma[np.isnan(self.corr_ma)] = 0

        corr_vec = np.abs(self.corr_ma[attr_num, :])
        corr_vec[corr_vec>=threshold] = 1
        return 1 - corr_vec

    def get_correlation_multi(self, attr_nums, threshold=1, coeffs=None):
        if self.corr_ma is None:
            lbls = np.load(self.label_file)
            self.corr_ma = np.corrcoef(lbls.transpose())
            self.corr_ma[np.isnan(self.corr_ma)] = 0

        if coeffs is not None:
            attr_nums = [a[0].item() for a in coeffs.nonzero()]

        corr_vec = np.abs(self.corr_ma[attr_nums, :])
        corr_vec[corr_vec>=threshold] = 1
        corr_vec = np.max(corr_vec, axis=0)

        return 1 - corr_vec

    def get_coeff(self, x):
        sign_0 = torch.relu(x-0.5).sign()
        sign_1 = torch.relu(0.5-x).sign()

        coeffs = sign_0*(-x) + sign_1*(1-x)

        attrs_idx = torch.randint(0, coeffs.shape[0], (1,), device=DEVICE)
        coeffs[torch.arange(coeffs.shape[0], device=DEVICE) != attrs_idx] = 0

        return coeffs

    def compute_eval_loss(self, w, coeff):
        self.w_0 = w
        predict_lbl_0 = self.Latent_Classifier(self.w_0.view(w.size(0), -1))
        lbl_0 = torch.sigmoid(predict_lbl_0)

        attr_pb_0 = lbl_0[torch.arange(lbl_0.shape[0]), self.attr_nums]

        target_pb = torch.clamp(attr_pb_0 + coeff, 0, 1).round()

        # Apply latent transformation
        self.w_1 = self.T_net(self.w_0.view(w.size(0), -1), coeff.unsqueeze(0), scaling=self.scaling)
        self.w_1 = self.w_1.view(w.size())
        predict_lbl_1 = self.Latent_Classifier(self.w_1.view(w.size(0), -1))

        # Pb loss
        self.loss_pb = self.BCEloss(predict_lbl_1[torch.arange(predict_lbl_1.shape[0]), self.attr_nums],
                                    target_pb, reduction='none')
        self.loss_pb = self.loss_pb.mean()

        # Latent code recon
        self.loss_recon = self.MSEloss(self.w_1, self.w_0)

        # Reg loss
        threshold_val = 1 if 'corr_threshold' not in self.config else self.config['corr_threshold']
        mask = torch.tensor(self.get_correlation(self.attr_nums[0], threshold=threshold_val)).type_as(predict_lbl_0)
        # mask = mask.repeat(predict_lbl_0.size(0), 1)  # We dont need to repeat the correlation mask in multi
        self.loss_reg = self.MSEloss(predict_lbl_1*mask, predict_lbl_0*mask)
        
        # Total loss
        w_recon, w_pb, w_reg = self.config['w']['recon'], self.config['w']['pb'], self.config['w']['reg']
        self.loss =  w_pb * self.loss_pb + w_recon*self.loss_recon + w_reg * self.loss_reg

        return self.loss.item(), w_pb*self.loss_pb.item(), w_recon*self.loss_recon.item(), w_reg*self.loss_reg.item()

    def compute_loss_multi(self, w, mask_input, n_iter):
        self.w_0 = w
        predict_lbl_0 = self.Latent_Classifier(self.w_0.view(w.size(0), -1))
        lbl_0 = torch.sigmoid(predict_lbl_0)

        attr_pb_0 = lbl_0[torch.arange(lbl_0.shape[0]), self.attr_nums]

        # Get scaling factor
        coeff = self.get_coeff(attr_pb_0)
        coeff_idx = coeff.nonzero(as_tuple=True)[0].tolist()

        target_pb = torch.clamp(attr_pb_0 + coeff, 0, 1).round()
        target_pb = target_pb[coeff_idx]

        # Apply latent transformation
        self.w_1 = self.T_net(self.w_0.view(w.size(0), -1), coeff.unsqueeze(0), scaling=self.scaling)
        self.w_1 = self.w_1.view(w.size())
        predict_lbl_1 = self.Latent_Classifier(self.w_1.view(w.size(0), -1))

        pred_pb = predict_lbl_1[torch.arange(predict_lbl_1.shape[0]), self.attr_nums]
        self.loss_pb = self.BCEloss(pred_pb[coeff_idx], target_pb, reduction='none')
        self.loss_pb = self.loss_pb.mean()

        # Latent code recon
        self.loss_recon = self.MSEloss(self.w_1, self.w_0)

        # Reg loss
        threshold_val = 1 if 'corr_threshold' not in self.config else self.config['corr_threshold']
        mask = torch.tensor(self.get_correlation_multi(self.attr_nums, threshold=threshold_val, coeffs=coeff)).type_as(predict_lbl_0)
        self.loss_reg = self.MSEloss(predict_lbl_1*mask, predict_lbl_0*mask)
        
        # Total loss
        w_recon, w_pb, w_reg = self.config['w']['recon'], self.config['w']['pb'], self.config['w']['reg']
        self.loss =  w_pb * self.loss_pb + w_recon*self.loss_recon + w_reg * self.loss_reg

        return self.loss
    
    def get_image(self, w):
        # Original image
        predict_lbl_0 = self.Latent_Classifier(w.view(w.size(0), -1))
        lbl_0 = torch.sigmoid(predict_lbl_0)

        self.local_attr = "multi"  # this is for saving purposes only

        attr_pb_0 = lbl_0[torch.arange(lbl_0.shape[0]), self.attr_nums]

        coeff = self.get_coeff(attr_pb_0)
        target_pb = torch.clamp(attr_pb_0 + coeff, 0, 1).round()

        if 'alpha' in self.config and not self.config['alpha']:
            coeff = 2 * target_pb.type_as(attr_pb_0) - 1 

        w_1 = self.T_net(w.view(w.size(0), -1), coeff.unsqueeze(0), scaling=self.scaling)
        w_1 = w_1.view(w.size())
        self.x_0, _ = self.StyleGAN([w], input_is_latent=True, randomize_noise=False)
        self.x_1, _ = self.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)

    def get_classification(self, w):
        predict_lbl_0 = self.Latent_Classifier(w.view(w.size(0), -1))
        return torch.sigmoid(predict_lbl_0).detach().cpu().numpy()[0]

    def get_original_image(self, w):
        x_0, _ = self.StyleGAN([w], input_is_latent=True, randomize_noise=False)
        return clip_img(downscale(x_0, 2))[0]

    def log_image(self, logger, w, n_iter):
        with torch.no_grad():
            self.get_image(w)
        logger.add_image('image_'+str(self.attrs)+'/iter'+str(n_iter+1)+f'_input_{self.local_attr}', clip_img(downscale(self.x_0, 2))[0], n_iter + 1)
        logger.add_image('image_'+str(self.attrs)+'/iter'+str(n_iter+1)+f'_modif_{self.local_attr}', clip_img(downscale(self.x_1, 2))[0], n_iter + 1)

    def log_loss(self, logger, n_iter):
        logger.add_scalar('loss_'+str(self.attrs)+'/class', self.loss_pb.item(), n_iter + 1)
        logger.add_scalar('loss_'+str(self.attrs)+'/latent_recon', self.loss_recon.item(), n_iter + 1)
        logger.add_scalar('loss_'+str(self.attrs)+'/attr_reg', self.loss_reg.item(), n_iter + 1)
        logger.add_scalar('loss_'+str(self.attrs)+'/total', self.loss.item(), n_iter + 1)

    def save_image(self, log_dir, w, n_iter):
        with torch.no_grad():
            self.get_image(w)
        utils.save_image(clip_img(self.x_0), log_dir + 'iter' +str(n_iter+1) + f'_input_{self.local_attr}' + '.jpg')
        utils.save_image(clip_img(self.x_1), log_dir + 'iter' +str(n_iter+1) + f'_modif_{self.local_attr}' + '.jpg')        

    def save_model(self, log_dir):
        torch.save(self.T_net.state_dict(),log_dir + '/tnet_' + str(self.attr_num) +'.pth.tar')

    def save_model_multi(self, log_dir, name=None):
        if name:
            filename = log_dir + '/tnet_' + name +'.pth.tar'
        else:
            filename = log_dir + '/tnet_' + "_".join(str(n) for n in self.attr_nums) +'.pth.tar'

        torch.save(self.T_net.state_dict(), filename)

    def save_checkpoint(self, n_epoch, log_dir):
        checkpoint_state = {
            'n_epoch': n_epoch,
            'T_net_state_dict': self.T_net.state_dict(),
            'opt_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        } 
        if (n_epoch+1) % 10 == 0 :
            torch.save(checkpoint_state, '{:s}/checkpoint'.format(log_dir)+'_'+str(n_epoch+1))
        else:
            torch.save(checkpoint_state, '{:s}/checkpoint'.format(log_dir))
    
    def load_model(self, log_dir):
        self.T_net.load_state_dict(torch.load(log_dir + 'tnet_' + str(self.attr_num) +'.pth.tar', map_location=DEVICE))

    def load_model_multi(self, log_dir, name=None):
        if name:
            filename = log_dir + '/tnet_' + name +'.pth.tar'
        else:
            filename = log_dir + '/tnet_' + "_".join(str(n) for n in self.attr_nums) +'.pth.tar'
        self.T_net.load_state_dict(torch.load(filename, map_location=DEVICE))

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        self.T_net.load_state_dict(state_dict['T_net_state_dict'])
        self.optimizer.load_state_dict(state_dict['opt_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        return state_dict['n_epoch'] + 1

    def update(self, w, mask, n_iter):
        self.n_iter = n_iter
        self.optimizer.zero_grad()
        self.compute_loss_multi(w, mask, n_iter).backward()
        self.optimizer.step()

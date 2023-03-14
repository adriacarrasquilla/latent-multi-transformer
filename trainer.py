# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from PIL import Image
from torch.autograd import grad
from torchvision import transforms, utils

from nets import *
from utils.functions import *

import sys 
sys.path.append('pixel2style2pixel/')

from pixel2style2pixel.models.stylegan2.model import Generator
from pixel2style2pixel.models.psp import get_keys

from attr_dict import NUM_TO_ATTR

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Trainer(nn.Module):
    def __init__(self, config, attr_num, attr, label_file):
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

        # self.attr_num = attr_num[0]
        # self.attr = attr[0]
        mapping_lrmul = self.config['mapping_lrmul']
        mapping_layers = self.config['mapping_layers']
        mapping_fmaps = self.config['mapping_fmaps']
        mapping_nonlinearity = self.config['mapping_nonlinearity']
        # Networks
        # Latent Transformer
        self.T_net = F_mapping_multi2(
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
        self.Latent_Classifier.load_state_dict(torch.load(classifier_model_path, map_location=device))
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
        # corr_vec = np.abs(self.corr_ma[attr_num:attr_num+1]) # Original, below is the experimental one
        # Enable this again for exp
        # attr_num = attr_num.cpu()
        corr_vec = np.abs(self.corr_ma[attr_num, :])
        corr_vec[corr_vec>=threshold] = 1
        return 1 - corr_vec

    def get_coeff(self, x):
        sign_0 = F.relu(x-0.5).sign()
        sign_1 = F.relu(0.5-x).sign()
        return sign_0*(-x) + sign_1*(1-x)

    def compute_loss(self, w, mask_input, n_iter):
        self.w_0 = w
        predict_lbl_0 = self.Latent_Classifier(self.w_0.view(w.size(0), -1))
        lbl_0 = torch.sigmoid(predict_lbl_0)
        attr_pb_0 = lbl_0[:, self.attr_num]

        # Get scaling factor
        coeff = self.get_coeff(attr_pb_0)
        target_pb = torch.clamp(attr_pb_0 + coeff, 0, 1).round()

        if 'alpha' in self.config and not self.config['alpha']:
            coeff = 2 * target_pb.type_as(attr_pb_0) - 1 
        # Apply latent transformation
        self.w_1 = self.T_net(self.w_0.view(w.size(0), -1), coeff)
        self.w_1 = self.w_1.view(w.size())
        predict_lbl_1 = self.Latent_Classifier(self.w_1.view(w.size(0), -1))

        # Pb loss
        T_coeff = target_pb.size(0)/(target_pb.sum(0) + 1e-8)
        F_coeff = target_pb.size(0)/(target_pb.size(0) - target_pb.sum(0) + 1e-8)

        mask_pb = T_coeff.float() * target_pb + F_coeff.float() * (1-target_pb)
        self.loss_pb = self.BCEloss(predict_lbl_1[:, self.attr_num], target_pb, reduction='none')*mask_pb
        self.loss_pb = self.loss_pb.mean()

        # Latent code recon
        self.loss_recon = self.MSEloss(self.w_1, self.w_0)

        # Reg loss
        threshold_val = 1 if 'corr_threshold' not in self.config else self.config['corr_threshold']
        mask = torch.tensor(self.get_correlation(self.attr_num, threshold=threshold_val)).type_as(predict_lbl_0)
        mask = mask.repeat(predict_lbl_0.size(0), 1)
        self.loss_reg = self.MSEloss(predict_lbl_1*mask, predict_lbl_0*mask)
        
        # Total loss
        w_recon, w_pb, w_reg = self.config['w']['recon'], self.config['w']['pb'], self.config['w']['reg']
        self.loss =  w_pb * self.loss_pb + w_recon*self.loss_recon + w_reg * self.loss_reg

        return self.loss

    def compute_loss_multi(self, w, mask_input, n_iter):
        self.w_0 = w
        predict_lbl_0 = self.Latent_Classifier(self.w_0.view(w.size(0), -1))
        lbl_0 = torch.sigmoid(predict_lbl_0)

        attr_pb_0 = lbl_0[torch.arange(lbl_0.shape[0]), self.attr_nums]

        # Get scaling factor
        coeff = self.get_coeff(attr_pb_0)
        # coeff = torch.tensor([-1,-1]).to(device)
        target_pb = torch.clamp(attr_pb_0 + coeff, 0, 1).round()

        if 'alpha' in self.config and not self.config['alpha']:
            coeff = 2 * target_pb.type_as(attr_pb_0) - 1 
            
        # coeff_scaled = 10 * coeff.type_as(attr_pb_0)

        # Apply latent transformation
        self.w_1 = self.T_net(self.w_0.view(w.size(0), -1), coeff.unsqueeze(0))
        # self.w_1 = self.T_net(self.w_0.view(w.size(0), -1), coeff_scaled.unsqueeze(0))
        self.w_1 = self.w_1.view(w.size())
        predict_lbl_1 = self.Latent_Classifier(self.w_1.view(w.size(0), -1))

        # Pb loss (experimental)
        # self.loss_pb = self.BCEloss(predict_lbl_1[torch.arange(predict_lbl_1.shape[0]), self.attr_nums],
        #                             target_pb, reduction='mean')
        # self.loss_pb = self.MultiLabelLoss(predict_lbl_1[torch.arange(predict_lbl_1.shape[0]), self.attr_nums],
        #                             target_pb, reduction='mean')

        # Original loss_pb down here
        # Pb loss
        T_coeff = target_pb.size(0)/(target_pb.sum(0) + 1e-8)
        F_coeff = target_pb.size(0)/(target_pb.size(0) - target_pb.sum(0) + 1e-8)
        mask_pb = T_coeff.float() * target_pb + F_coeff.float() * (1-target_pb)
        self.loss_pb = self.BCEloss(predict_lbl_1[torch.arange(predict_lbl_1.shape[0]), self.attr_nums],
                                    target_pb, reduction='none')*mask_pb
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

        return self.loss

    def compute_loss_multi_experiment(self, w, mask_input, n_iter):
        self.w_0 = w
        predict_lbl_0 = self.Latent_Classifier(self.w_0.view(w.size(0), -1))
        lbl_0 = torch.sigmoid(predict_lbl_0)
        self.attr_num = torch.argmax(lbl_0, axis=1)
        attr_pb_0 = lbl_0[torch.arange(lbl_0.shape[0]), self.attr_num]

        # Get scaling factor
        coeff = self.get_coeff(attr_pb_0)
        target_pb = torch.clamp(attr_pb_0 + coeff, 0, 1).round()

        if 'alpha' in self.config and not self.config['alpha']:
            coeff = 2 * target_pb.type_as(attr_pb_0) - 1 

        # Apply latent transformation
        self.w_1 = self.T_net(self.w_0.view(w.size(0), -1), coeff)
        self.w_1 = self.w_1.view(w.size())
        predict_lbl_1 = self.Latent_Classifier(self.w_1.view(w.size(0), -1))

        # Pb loss
        T_coeff = target_pb.size(0)/(target_pb.sum(0) + 1e-8)
        F_coeff = target_pb.size(0)/(target_pb.size(0) - target_pb.sum(0) + 1e-8)

        mask_pb = T_coeff.float() * target_pb + F_coeff.float() * (1-target_pb)
        self.loss_pb = self.BCEloss(predict_lbl_1[torch.arange(predict_lbl_1.shape[0]), self.attr_num],
                                    target_pb, reduction='none')*mask_pb
        self.loss_pb = self.loss_pb.mean()

        # Latent code recon
        self.loss_recon = self.MSEloss(self.w_1, self.w_0)

        # Reg loss
        threshold_val = 1 if 'corr_threshold' not in self.config else self.config['corr_threshold']
        mask = torch.tensor(self.get_correlation(self.attr_num, threshold=threshold_val)).type_as(predict_lbl_0)
        # mask = mask.repeat(predict_lbl_0.size(0), 1)  # We dont need to repeat the correlation mask in multi
        self.loss_reg = self.MSEloss(predict_lbl_1*mask, predict_lbl_0*mask)
        
        # Total loss
        w_recon, w_pb, w_reg = self.config['w']['recon'], self.config['w']['pb'], self.config['w']['reg']
        self.loss =  w_pb * self.loss_pb + w_recon*self.loss_recon + w_reg * self.loss_reg

        return self.loss
    
    def get_image(self, w):
        # Original image
        predict_lbl_0 = self.Latent_Classifier(w.view(w.size(0), -1))
        lbl_0 = torch.sigmoid(predict_lbl_0)
        # attr_pb_0 = lbl_0[:, self.attr_num] # TODO: find a global way to do this, or do separate functions
        # self.attr_num = torch.argmax(lbl_0, axis=1)
        # self.local_attr = NUM_TO_ATTR[self.attr_num.item()]
        self.local_attr = "Bald"
        # attr_pb_1 = lbl_0[torch.arange(lbl_0.shape[0]), self.attr_num]
        attr_pb_0 = lbl_0[torch.arange(lbl_0.shape[0]), self.attr_nums]

        coeff = self.get_coeff(attr_pb_0)
        target_pb = torch.clamp(attr_pb_0 + coeff, 0, 1).round()

        if 'alpha' in self.config and not self.config['alpha']:
            coeff = 2 * target_pb.type_as(attr_pb_0) - 1 

        w_1 = self.T_net(w.view(w.size(0), -1), coeff.unsqueeze(0))
        w_1 = w_1.view(w.size())
        self.x_0, _ = self.StyleGAN([w], input_is_latent=True, randomize_noise=False)
        self.x_1, _ = self.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)

    def log_image(self, logger, w, n_iter):
        with torch.no_grad():
            self.get_image(w)
        logger.add_image('image_'+self.attr+'/iter'+str(n_iter+1)+f'_input_{self.local_attr}', clip_img(downscale(self.x_0, 2))[0], n_iter + 1)
        logger.add_image('image_'+self.attr+'/iter'+str(n_iter+1)+f'_modif_{self.local_attr}', clip_img(downscale(self.x_1, 2))[0], n_iter + 1)

    def get_image_verbose(self, w):
        # Original image
        predict_lbl_0 = self.Latent_Classifier(w.view(w.size(0), -1))
        lbl_0 = torch.sigmoid(predict_lbl_0)
        # attr_pb_0 = lbl_0[:, self.attr_num] # TODO: find a global way to do this, or do separate functions
        self.attr_num = torch.argmax(lbl_0, axis=1)
        self.local_attr = NUM_TO_ATTR[self.attr_num.item()]
        # attr_pb_1 = lbl_0[torch.arange(lbl_0.shape[0]), self.attr_num]
        attr_pb_0 = lbl_0[torch.arange(lbl_0.shape[0]), self.attr_nums]

        coeff = self.get_coeff(attr_pb_0)
        target_pb = torch.clamp(attr_pb_0 + coeff, 0, 1).round()

        if 'alpha' in self.config and not self.config['alpha']:
            coeff = 2 * target_pb.type_as(attr_pb_0) - 1 

        w_1 = self.T_net(w.view(w.size(0), -1), coeff.unsqueeze(0))
        w_1 = w_1.view(w.size())
        self.x_0, _ = self.StyleGAN([w], input_is_latent=True, randomize_noise=False)
        self.x_1, _ = self.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)

        predict_lbl_1 = self.Latent_Classifier(w_1.view(w.size(0), -1))

        predict = predict_lbl_1[torch.arange(predict_lbl_1.shape[0]), self.attr_nums]

        return attr_pb_0, target_pb, predict

    def log_image_verbose(self, logger, w, n_iter):
        with torch.no_grad():
            org, target, predict = self.get_image_verbose(w)
        suffix = ""
        for o, t, p in zip(org, target, predict):
            suffix += f",org_{o.item():.2f},target_{t.item():.2f},pred_{p.item():.2f}"
        # logger.add_image('image_'+self.attr+'/iter'+str(n_iter+1)+f'_input_{self.local_attr}', clip_img(downscale(self.x_0, 2))[0], n_iter + 1)
        # logger.add_image('image_'+self.attr+'/iter'+str(n_iter+1)+f'_modif_{self.local_attr}', clip_img(downscale(self.x_1, 2))[0], n_iter + 1)
        logger.add_image('image_'+str(self.attrs)+'/iter'+str(n_iter+1)+'input', clip_img(downscale(self.x_0, 2))[0], n_iter + 1)
        logger.add_image('image_'+str(self.attrs)+'/iter'+str(n_iter+1)+'modif'+suffix, clip_img(downscale(self.x_1, 2))[0], n_iter + 1)
        
    def log_loss(self, logger, n_iter):
        logger.add_scalar('loss_'+self.attr+'/class', self.loss_pb.item(), n_iter + 1)
        logger.add_scalar('loss_'+self.attr+'/latent_recon', self.loss_recon.item(), n_iter + 1)
        logger.add_scalar('loss_'+self.attr+'/attr_reg', self.loss_reg.item(), n_iter + 1)
        logger.add_scalar('loss_'+self.attr+'/total', self.loss.item(), n_iter + 1)

    def save_image(self, log_dir, w, n_iter):
        with torch.no_grad():
            self.get_image(w)
        utils.save_image(clip_img(self.x_0), log_dir + 'iter' +str(n_iter+1) + f'_input_{self.local_attr}' + '.jpg')
        utils.save_image(clip_img(self.x_1), log_dir + 'iter' +str(n_iter+1) + f'_modif_{self.local_attr}' + '.jpg')        

    def save_model(self, log_dir):
        torch.save(self.T_net.state_dict(),log_dir + '/tnet_' + str(self.attr_num) +'.pth.tar')

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
        self.T_net.load_state_dict(torch.load(log_dir + 'tnet_' + str(self.attr_num) +'.pth.tar', map_location=device))

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.T_net.load_state_dict(state_dict['T_net_state_dict'])
        self.optimizer.load_state_dict(state_dict['opt_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        return state_dict['n_epoch'] + 1

    def update(self, w, mask, n_iter):
        self.n_iter = n_iter
        self.optimizer.zero_grad()
        self.compute_loss_multi(w, mask, n_iter).backward()
        self.optimizer.step()
        

import os
from glob import glob
import sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
import logging
import random

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

from utils import *
from data import MRI_7t, save_nii
from metrics import cal_psnr_ssim_list
from teacher import TeacherEncoder, TeacherDecoder


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, pad=1, dropout=False):
        super().__init__()
        layers = [
            nn.Conv3d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, bias=False),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        # layers.extend([
        #     nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.InstanceNorm3d(out_c),
        #     nn.LeakyReLU(inplace=True)
        # ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class DeconvMerge(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, pad=1):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(inplace=True)
        )
        self.conv = ConvBlock(out_c*2, out_c, kernel=kernel, pad=pad)

    def forward(self, x, skip_x):
        x = self.deconv(x)
        if x.shape[2:] != skip_x.shape[2:]:
            x = match_size(x, skip_x.shape[2:])
        x = torch.cat((x, skip_x), dim=1)
        x = self.conv(x)
        return x


class UpMerge(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, pad=1):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.conv = ConvBlock(out_c*2, out_c, kernel=kernel, pad=pad)

    def forward(self, x, skip_x):
        x = F.upsample(x, size=skip_x.shape[2:], mode='trilinear')
        x = self.conv1x1(x)
        x = torch.cat((x, skip_x), dim=1)
        x = self.conv(x)
        return x


# https://github.com/13952522076/PRM/blob/master/models/mnasnet_prm.py
class PRMLayer(nn.Module):
    def __init__(self, mode='dotproduct'):
        super(PRMLayer, self).__init__()
        self.mode = mode
        self.max_pool = nn.AdaptiveMaxPool3d(1,return_indices=True)
        self.weight = Parameter(torch.zeros(1,1,1,1))
        self.bias = Parameter(torch.ones(1,1,1,1))
        self.sig = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.one = Parameter(torch.ones(1,1))
        self.zero = Parameter(torch.zeros(1,1))
        self.theta = Parameter(torch.rand(1,3,1,1,1))
        self.scale = Parameter(torch.ones(1))

    def forward(self, x):
        b, c, h, w, d = x.shape
        position_mask = self.get_position_mask(x, b, c, h, w, d) # Output (b, 3, h, w, d)

        # Similarity function
        query_value, query_position = self.get_query_position(x, b, c, h, w, d)
        query_value = query_value.view(b, -1, 1)
        x_value = x.view(b, -1, h*w*d)
        similarity_max = self.get_similarity(x_value, query_value, mode=self.mode)
        similarity_gap = self.get_similarity(x_value, self.gap(x).view(b, -1, 1), mode=self.mode)
        # similarity_max = similarity_max.view(b, h*w*d)

        Distance = abs(position_mask - query_position)
        Distance = Distance.type(query_value.type())
        distribution = Normal(0, self.scale)
        Distance = distribution.log_prob(Distance * self.theta).exp().clone()
        Distance = (Distance.mean(dim=1)).view(b, h*w*d)
        similarity_max = similarity_max * Distance

        # similarity_gap = similarity_gap.view(b, h*w*d)
        similarity = similarity_max*self.zero+similarity_gap*self.one

        context = similarity - similarity.mean(dim=1, keepdim=True)
        std = context.std(dim=1, keepdim=True) + 1e-5
        context = (context/std).view(b, h, w, d)
        # affine function
        context = context * self.weight + self.bias
        context = context.view(b, 1, h, w, d).expand(b, c, h, w, d).reshape(b, c, h, w, d)
        value = x*self.sig(context)

        return value

    def get_position_mask(self, x, b, c, h, w, d):
        mask = (torch.ones((h, w, d))).nonzero().cuda()
        mask = (mask.reshape(h, w, d, 3)).permute(3, 0, 1, 2).expand(b, 3, h, w, d)
        return mask

    def get_query_position(self, x, b, c, h, w, d):
        sumvalue = x.sum(dim=1, keepdim=True)
        maxvalue, maxposition = self.max_pool(sumvalue)
        t_position = torch.cat((maxposition//w//d, maxposition//d//h, maxposition//h//w), dim=1)

        t_value = x[torch.arange(b),:,t_position[:,0,0,0,0],t_position[:,1,0,0,0],t_position[:,2,0,0,0]]
        # t_value = t_value.view(b, c, 1, 1, 1)
        return t_value, t_position

    def get_similarity(self, query, key_value, mode='dotproduct'):
        if mode == 'dotproduct':
            similarity = torch.matmul(key_value.permute(0, 2, 1), query).squeeze(dim=1)
        elif mode == 'l1norm':
            similarity = -(abs(query - key_value)).sum(dim=1)
        elif mode == 'gaussian':
            # Gaussian Similarity (No recommanded, too sensitive to noise)
            similarity = torch.exp(torch.matmul(key_value.permute(0, 2, 1), query))
            similarity[similarity == float("Inf")] = 0
            similarity[similarity <= 1e-9] = 1e-9
        elif mode == 'cosine':
            cos = nn.CosineSimilarity(dim=1)
            similarity = cos(query, key_value)
        else:
            similarity = torch.matmul(key_value.permute(0, 2, 1), query)
        return similarity


class GuideBlock(nn.Module):
    def __init__(self, in_c_x, in_c_skip, out_c, kernel=3, pad=1):
        super().__init__()
        self.conv1x1_x = nn.Sequential(
            nn.Conv3d(in_c_x, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.conv_skip = ConvBlock(in_c_skip, out_c, kernel=kernel, pad=pad)

        self.attention = nn.Sequential(
            nn.Conv3d(out_c*2, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.prm = PRMLayer()
        # self.prm = PRMLayer(mode='cosine')

    def forward(self, x, skip_x):
        b, c, h, w, d = skip_x.shape

        x = F.upsample(x, size=(h, w, d), mode='trilinear')
        x = self.conv1x1_x(x)
        skip_x = self.conv_skip(skip_x)

        att = torch.cat((x, skip_x), dim=1)
        att = self.attention(att)
        x = x * att[:, 0].view(b, 1, h, w, d) + skip_x * att[:, 1].view(b, 1, h, w, d)

        x = self.prm(x)

        return x


class FeatureExtractionBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, pad): # out_c => growth rate
        super().__init__()
        self.conv1 = ConvBlock(in_c, out_c, kernel=kernel, pad=pad)
        self.conv2 = ConvBlock(in_c+out_c, out_c, kernel=kernel, pad=pad)
        self.conv3 = ConvBlock(in_c+out_c*2, out_c, kernel=kernel, pad=pad)
        # self.conv4 = ConvBlock(in_c+out_c*3, out_c, kernel=kernel, pad=pad)

    def forward(self, x):
        c1 = self.conv1(x)

        c2 = torch.cat([x, c1], dim=1)
        c2 = self.conv2(c2)

        c3 = torch.cat([x, c1, c2], dim=1)
        c3 = self.conv3(c3)

        c4 = torch.cat([x, c1, c2, c3], dim=1)
        # c4 = self.conv4(c4)
        #
        # out = torch.cat([x, c1, c2, c3, c4], dim=1)

        return c4


class KnowledgeKeeperNet(nn.Module):
    def __init__(self, args, in_c=1):
        super().__init__()
        self.args = args
        nf = self.args.nf
        nf_down = self.args.nf // 4

        extract_k, extract_p = 7, 3
        transition_k, transition_s, transition_p = 4, 2, 1
        guide_k, guide_p = 7, 3

        num_conv = 3 # the number of ConvBlock in DenseBlock

        self.conv1 = FeatureExtractionBlock(in_c, nf_down, kernel=extract_k, pad=extract_p)
        self.conv2 = FeatureExtractionBlock(nf_down, nf_down*2, kernel=extract_k, pad=extract_p)
        self.conv3 = FeatureExtractionBlock(nf_down*2, nf_down*4, kernel=extract_k, pad=extract_p)
        self.conv4 = FeatureExtractionBlock(nf_down*4, nf_down*8, kernel=extract_k, pad=extract_p)
        self.conv5 = FeatureExtractionBlock(nf_down*8, nf_down*16, kernel=extract_k, pad=extract_p)

        self.tran1 = ConvBlock(in_c+nf_down*num_conv, nf_down, kernel=transition_k, stride=transition_s, pad=transition_p)
        self.tran2 = ConvBlock(nf_down+nf_down*2*num_conv, nf_down*2, kernel=transition_k, stride=transition_s, pad=transition_p)
        self.tran3 = ConvBlock(nf_down*2+nf_down*4*num_conv, nf_down*4, kernel=transition_k, stride=transition_s, pad=transition_p)
        self.tran4 = ConvBlock(nf_down*4+nf_down*8*num_conv, nf_down*8, kernel=transition_k, stride=transition_s, pad=transition_p)

        self.guide5_conv = ConvBlock(nf_down*8+nf_down*16*num_conv, nf*16, kernel=guide_k, pad=guide_p)
        self.guide5_PRM = PRMLayer()
        self.guide4 = GuideBlock(nf*16, nf_down*4+nf_down*8*num_conv, nf*8, kernel=guide_k, pad=guide_p)
        self.guide3 = GuideBlock(nf*8, nf_down*2+nf_down*4*num_conv, nf*4, kernel=guide_k, pad=guide_p)
        self.guide2 = GuideBlock(nf*4, nf_down+nf_down*2*num_conv, nf*2, kernel=guide_k, pad=guide_p)
        self.guide1 = GuideBlock(nf*2, in_c+nf_down*num_conv, nf, kernel=guide_k, pad=guide_p)

    def forward(self, x):
        c1 = self.conv1(x)
        t1 = self.tran1(c1)

        c2 = self.conv2(t1)
        t2 = self.tran2(c2)

        c3 = self.conv3(t2)
        t3 = self.tran3(c3)

        c4 = self.conv4(t3)
        t4 = self.tran4(c4)

        c5 = self.conv5(t4)

        g5 = self.guide5_conv(c5)
        g5 = self.guide5_PRM(g5)
        g4 = self.guide4(g5, c4)
        g3 = self.guide3(g4, c3)
        g2 = self.guide2(g3, c2)
        g1 = self.guide1(g2, c1)

        enc_list = [g1, g2, g3, g4, g5]

        return enc_list


class Discriminator(nn.Module):
    def __init__(self, args, in_c=1):
        super().__init__()
        self.args = args
        nf = self.args.nf

        self.conv1 = ConvBlock(in_c, nf, kernel=4, stride=2, pad=1)
        self.conv2 = ConvBlock(nf, nf*2, kernel=4, stride=2, pad=1)
        self.conv3 = ConvBlock(nf*2, nf*4, kernel=4, stride=2, pad=1)
        self.conv4 = ConvBlock(nf*4, nf*8, kernel=4, stride=2, pad=1)
        self.out = nn.Sequential(
            nn.Conv3d(nf*8, 1, kernel_size=4, padding=1, bias=False)
            # nn.AdaptiveAvgPool3d(output_size=1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        out = self.out(c4)

        return out


class FusionModules(nn.Module):
    def __init__(self, args, guided_num=5):
        super().__init__()
        # self.args = args
        self.alphas = Parameter(torch.tensor([.5]*guided_num))
        self.betas = Parameter(torch.tensor([.5]*guided_num))
        self.relu = nn.ReLU()

    def forward(self, guide_feat, enc_feat, levels=[0,1,2,3,4]):
        alphas = self.relu(self.alphas)
        betas = self.relu(self.betas)

        out = []
        assert len(alphas) == len(levels), 'the number of levels should be same the number of guidance'
        for lev, guide in enumerate(guide_feat):
            if lev in levels:
                g = torch.mul(guide_feat[lev], alphas[lev])
                e = torch.mul(enc_feat[lev], betas[lev])
                out.append(g+e)
            else:
                out.append(enc_feat[lev])
        return out


class Implementation(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.path_dataset = self.args.path_dataset_7T
        self.path_log = self.args.path_log
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.lr = self.args.lr
        self.checkpoint_sample = self.args.checkpoint_sample
        self.stop_patience = self.args.stop_patience

        self.lambda_adv = self.args.lambda_adv
        self.lambda_vox = self.args.lambda_vox
        self.datetime_teacher = self.args.datetime_teacher

    def training(self, device, fold, datetime_train):

        fold_name = 'Fold_%02d' % fold
        val_idx = [fold]

        ##### Directory
        dir_log = f'{self.path_log}/{self.datetime_teacher}_T_{datetime_train}_K'
        dir_model = f'{dir_log}/model/{fold_name}'
        dir_tboard = f'{dir_log}/tboard/{fold_name}'
        dir_result = f'{dir_log}/result_valid/{fold_name}'

        directory = [dir_log, dir_model, dir_tboard, dir_result]
        for dir in directory:
            os.makedirs(dir, exist_ok=True)

        ##### Training Log
        logger, stream_handler, file_handler = logger_setting(file_name=f'{dir_log}/log_{fold_name}.log')
        logger.debug('============================================')
        logger.debug(f'Fold Name: {fold_name}')
        logger.debug('Batch Size: %d' % self.batch_size)
        logger.debug('Epoch: %d' % self.epochs)

        writer = SummaryWriter(dir_tboard)

        ##### Dataset Load
        train_data_path = []
        val_data_path = []
        for folder_name in sorted(os.listdir(self.path_dataset)):
            _, patient_id = folder_name.split('_')  # folder_name example: S_01
            if int(patient_id) in val_idx:
                val_data_path.append(f'{self.path_dataset}/{folder_name}')
            else:
                train_data_path.append(f'{self.path_dataset}/{folder_name}')

        logger.info(f'''Valid data: {[path.split('/')[-1] for path in val_data_path]}''')
        logger.info(f'''Train data: {[path.split('/')[-1] for path in train_data_path]}''')

        train_dataset = MRI_7t(train_data_path, train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = MRI_7t(val_data_path)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        ##### Initialize
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        loss_L1 = nn.L1Loss()
        loss_MSE = nn.MSELoss()

        ##### Model
        keeper = nn.DataParallel(KnowledgeKeeperNet(self.args)).to(device)
        discriminator = nn.DataParallel(Discriminator(self.args)).to(device)

        keeper.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

        optimizer_K = torch.optim.Adam(keeper.parameters(), lr=self.lr, betas=(0.9, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # scheduler_K = torch.optim.lr_scheduler.StepLR(optimizer_K, step_size=1, gamma=0.99)
        # scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=1, gamma=0.99)


        ##### Pretrained model (Teacher)
        pretrain_path = f'{self.path_log}/{self.datetime_teacher}_T/model/{fold_name}'

        teacher_enc = nn.DataParallel(TeacherEncoder(self.args)).to(device)
        teacher_dec = nn.DataParallel(TeacherDecoder(self.args)).to(device)

        teacher_enc.load_state_dict(torch.load(pretrain_path + '/teacher_encoder.pth'))
        teacher_dec.load_state_dict(torch.load(pretrain_path + '/teacher_decoder.pth'))

        for param in teacher_enc.parameters():
            param.requires_grad = False
        for param in teacher_dec.parameters():
            param.requires_grad = False

        logger.debug('============================================')
        logger.debug('[Teacher]')
        logger.debug(f'Pretrained model: {pretrain_path}')
        logger.debug('[Knowledge keeper]')
        logger.debug(str(keeper))
        logger.debug('[Discriminator]')
        logger.debug(str(discriminator))
        logger.debug('Learning Rate: %.5f' % self.lr)
        logger.debug('Lambda for voxel-wise loss: %d' % self.lambda_vox)
        logger.debug('Lambda for adversarial loss: %d' % self.lambda_adv)
        logger.debug('============================================')

        ##### Training
        best = {'epoch': 0, 'psnr': 0, 'ssim': 0}
        patience = 0

        for epoch in tqdm(range(1, self.epochs + 1), desc='Epoch'):
            update = False
            train_loss = {'loss_D_total': 0, 'loss_K_total': 0, 'loss_K_distill_L2': 0, 'loss_K_distill_match': 0, 'loss_K_adv': 0, 'loss_K_vox': 0}
            keeper.train()
            discriminator.train()

            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Batch'):
                # logger.debug(f'[Epoch: {epoch} | Batch No.: {i}]')
                real_x = Variable(batch['x']).to(device)
                real_y = Variable(batch['y']).to(device)

                # ------------------------------
                # Discriminator
                # ------------------------------

                pred_y = teacher_dec(keeper(real_x))

                # Real
                pred_real = discriminator(real_y)
                valid = Variable(Tensor(np.ones(pred_real.size())), requires_grad=False)
                loss_D_real = loss_MSE(pred_real, valid)

                # Fake (generated)
                pred_fake = discriminator(pred_y.detach())
                fake = Variable(Tensor(np.zeros(pred_fake.size())), requires_grad=False)
                loss_D_fake = loss_MSE(pred_fake, fake)

                # Total loss
                loss_D = self.lambda_adv * (loss_D_real + loss_D_fake)

                # logger.debug(
                #     f'[Discriminator] pred_real (mean): {round(torch.mean(pred_real).item(), 4)} | pred_fake (mean): {round(torch.mean(pred_fake).item(), 4)} | Loss: {round(loss_D.item(), 4)}'
                # )

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

                del pred_y, pred_real, pred_fake

                # ------------------------------
                # Keeper
                # ------------------------------

                # Distillation loss
                teach_1, teach_2, teach_3, teach_4, teach_5 = teacher_enc(real_y) # feature maps of teacher encoder
                guided_1, guided_2, guided_3, guided_4, guided_5 = keeper(real_x) # feature maps of knowledge keeper

                loss_K_distill_L2 = loss_MSE(guided_1, teach_1) + loss_MSE(guided_2, teach_2) + loss_MSE(guided_3, teach_3) + loss_MSE(guided_4, teach_4) + loss_MSE(guided_5, teach_5)
                loss_K_distill_match = loss_dist_match(guided_1, teach_1) + loss_dist_match(guided_2, teach_2) + loss_dist_match(guided_3,teach_3) + loss_dist_match(guided_4, teach_4) + loss_dist_match(guided_5, teach_5)

                # Voxel-wise loss
                fake_y = teacher_dec([guided_1, guided_2, guided_3, guided_4, guided_5])
                loss_K_vox = self.lambda_vox * loss_L1(fake_y, real_y)

                # Adversarial loss
                pred_fake = discriminator(fake_y)
                loss_K_adv = self.lambda_adv * loss_MSE(pred_fake, valid)

                # Total loss
                loss_K = loss_K_distill_L2 + loss_K_distill_match + loss_K_vox + loss_K_adv

                # logger.debug(
                #     f'[Generator] pred_fake (mean): {round(torch.mean(pred_fake).item(), 4)} | Adv Loss: {round(loss_G_fake.item(), 4)} | Total Loss: {round(loss_G.item(), 4)}'
                # )

                optimizer_K.zero_grad()
                loss_K.backward()
                optimizer_K.step()

                del teach_1, teach_2, teach_3, teach_4, teach_5
                del guided_1, guided_2, guided_3, guided_4, guided_5
                del fake_y, pred_fake

                train_loss['loss_D_total'] += loss_D.item()
                train_loss['loss_K_total'] += loss_K.item()
                train_loss['loss_K_distill_L2'] += loss_K_distill_L2.item()
                train_loss['loss_K_distill_match'] += loss_K_distill_match.item()
                train_loss['loss_K_vox'] += loss_K_vox.item()
                train_loss['loss_K_adv'] += loss_K_adv.item()

            if epoch % self.checkpoint_sample == 0:
                real_y_list, pred_y_list, _ = self.prediction(val_dataloader, keeper, teacher_dec, device,
                                                              save_pred_path=f'{dir_result}/{epoch}_')
            else:
                real_y_list, pred_y_list, _ = self.prediction(val_dataloader, keeper, teacher_dec, device)

            val_psnr, val_ssim = cal_psnr_ssim_list(real_y_list, pred_y_list)
            loss_D_ = train_loss['loss_D_total'] / len(train_dataloader)
            loss_K_ = train_loss['loss_K_total'] / len(train_dataloader)
            loss_K_distill_match_ = train_loss['loss_K_distill_match'] / len(train_dataloader)
            loss_K_distill_L2_ = train_loss['loss_K_distill_L2'] / len(train_dataloader)
            loss_K_vox_ = train_loss['loss_K_vox'] / len(train_dataloader)
            loss_K_adv_ = train_loss['loss_K_adv'] / len(train_dataloader)
            if best['psnr'] < val_psnr and best['ssim'] < val_ssim:
                torch.save(keeper.state_dict(), f'{dir_model}/knowledge_keeper.pth')
                torch.save(discriminator.state_dict(), f'{dir_model}/discriminator.pth')
                best['epoch'] = epoch
                best['psnr'] = val_psnr
                best['ssim'] = val_ssim
                update = True
                patience = 0
            else:
                patience += 1

            logger.info(f'[Epoch: {epoch}/{self.epochs}]')
            # logger.info(f'[Epoch: {epoch}/{args.epochs}] lr: {scheduler_K.get_last_lr()}')
            logger.info(
                f'D loss: {round(loss_D_, 4)} | K loss: {round(loss_K_, 4)} | distill_L2: {round(loss_K_distill_L2_, 4)} | distill_match: {round(loss_K_distill_match_, 4)} | vox: {round(loss_K_vox_, 4)} | adv: {round(loss_K_adv_, 4)} | val_psnr: {round(val_psnr, 4)} | val_ssim: {round(val_ssim, 4)} | valid_update: {str(update)}({patience})'
            )

            writer.add_scalar('loss_D', loss_D_, epoch)
            writer.add_scalar('loss_K', loss_K_, epoch)
            writer.add_scalar('loss_K_distill_L2', loss_K_distill_L2_, epoch)
            writer.add_scalar('loss_K_distill_match', loss_K_distill_match_, epoch)
            writer.add_scalar('loss_K_vox', loss_K_vox_, epoch)
            writer.add_scalar('loss_K_adv', loss_K_adv_, epoch)
            writer.add_scalar('val_psnr', val_psnr, epoch)
            writer.add_scalar('val_ssim', val_ssim, epoch)

            if patience == self.stop_patience:
                logger.info(
                    f'-------------------------------------------- Early Stopping ! Patience: {self.stop_patience}')
                break

            # scheduler_K.step()
            # scheduler_D.step()

        writer.close()

        logger.info('============================================')
        logger.info(f'[Best Performance for Validation]')
        logger.info(
            f'''Epoch: {best['epoch']} | PSNR: {best['psnr']} | SSIM: {best['ssim']}''')
        logger_closing(logger, stream_handler, file_handler)

        del keeper, discriminator

        ##### Load best model
        keeper_dict = torch.load(f'{dir_model}/knowledge_keeper.pth')
        keeper = nn.DataParallel(KnowledgeKeeperNet(self.args)).to(device)
        keeper.load_state_dict(keeper_dict)

        ##### Testing
        self.testing(device, datetime_train, save_output=True)
        dir_all = f'{dir_log}/result_all/'
        os.makedirs(dir_all, exist_ok=True)
        logger_all, stream_handler_all, file_handler_all = logger_setting(file_name=f'{dir_all}/log_all.log')
        real_y_list, pred_y_list, patient_ids = self.prediction(val_dataloader, keeper, teacher_dec, device,
                                                      save_pred_path=dir_all)
        mean_psnr, mean_ssim, total_psnr, total_ssim = cal_psnr_ssim_list(real_y_list, pred_y_list, return_total=True)

        for idx, patient_id in enumerate(patient_ids):
            logger_all.info(f'{fold_name} | {patient_id} | {total_psnr[idx]} | {total_ssim[idx]}')
        logger_closing(logger_all, stream_handler_all, file_handler_all)

        logger, stream_handler, file_handler = logger_setting(file_name=f'{dir_log}/log_{fold_name}.log')
        logger.info('[Test Result]')
        logger.info(f'PSNR: {round(mean_psnr,4)} | SSIM: {round(mean_ssim,4)}')
        logger_closing(logger, stream_handler, file_handler)

        torch.cuda.empty_cache()


    def testing(self, device, datetime_train, save_output=True):
        dir_log = f'{self.path_log}/{self.datetime_teacher}_T_{datetime_train}_K'
        dir_all = f'{dir_log}/result_all/'
        os.makedirs(dir_all, exist_ok=True)

        logger_all, stream_handler_all, file_handler_all = logger_setting(file_name=f'{dir_all}/log_all.log')
        logger_all.info(f'Model: {datetime_train}')

        all_psnr = []
        all_ssim = []
        logger_all.info('[Fold | Patient ID | PSNR | SSIM]')

        fold_names = sorted(os.listdir(f'{dir_log}/model'))
        # for fold in range(1, 16):
        for fold_name in fold_names:
            dir_pretrain = f'{self.path_log}/{self.datetime_teacher}_T/model/{fold_name}'
            teacher_dec_dict = torch.load(f'{dir_pretrain}/teacher_decoder.pth')
            teacher_dec = nn.DataParallel(TeacherDecoder(self.args)).to(device)
            teacher_dec.load_state_dict(teacher_dec_dict)

            dir_model = f'{dir_log}/model/{fold_name}'
            keeper_dict = torch.load(f'{dir_model}/knowledge_keeper.pth')
            keeper = nn.DataParallel(KnowledgeKeeperNet(self.args)).to(device)
            keeper.load_state_dict(keeper_dict)

            data_path = [f'{self.path_dataset}/S_{fold_name[-2:]}']
            dataset = MRI_7t(data_path)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            if save_output:
                real_y, pred_y, patient_ids = self.prediction(dataloader, keeper, teacher_dec, device, dir_all)
            else:
                real_y, pred_y, patient_ids = self.prediction(dataloader, keeper, teacher_dec, device)

            mean_psnr, mean_ssim, total_psnr, total_ssim = cal_psnr_ssim_list(real_y, pred_y, return_total=True)

            for idx, patient_id in enumerate(patient_ids):
                logger_all.info(f'{fold_name} | {patient_id} | {total_psnr[idx]} | {total_ssim[idx]}')

            all_psnr.append(mean_psnr)
            all_ssim.append(mean_ssim)

        all_psnr = np.array(all_psnr)
        all_ssim = np.array(all_ssim)
        np.save(f'{dir_all}total_psnr', all_psnr)
        np.save(f'{dir_all}total_ssim', all_ssim)

        logger_all.info('[Test Result]')
        logger_all.info(f'PSNR: {all_psnr.mean()} | SSIM: {all_ssim.mean()}')

        stream_handler_all.close()
        logger_all.removeHandler(stream_handler_all)
        file_handler_all.close()
        logger_all.removeHandler(file_handler_all)
        del logger_all, stream_handler_all, file_handler_all


    def prediction(self, dataloader, keeper, teacher_decoder, device, save_pred_path=False):
        patient_ids = []
        real_y_list = []
        pred_y_list = []

        keeper.eval()
        with torch.no_grad():
            for batch in dataloader:
                real_x = Variable(batch['x']).to(device)
                real_y = Variable(batch['y']).to(device)
                pred_y = teacher_decoder(keeper(real_x))

                real_y = real_y.cpu().detach().numpy()
                pred_y = pred_y.cpu().detach().numpy()

                for idx in range(pred_y.shape[0]):
                    patient_id = str(batch['patient_id'][idx])
                    patient_ids.append(patient_id)

                    real_y_ = real_y[idx].squeeze()
                    pred_y_ = pred_y[idx].squeeze()

                    real_y_list.append(real_y_)
                    pred_y_list.append(pred_y_)

                    if save_pred_path:
                        save_nii(pred_y_, f'{save_pred_path}{patient_id}_pred_y')

        return real_y_list, pred_y_list, patient_ids

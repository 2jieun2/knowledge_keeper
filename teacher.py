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

from utils import *
from data import MRI_7t, save_nii
from metrics import cal_psnr_ssim_list


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


class TeacherEncoder(nn.Module):
    def __init__(self, args, in_c=1):
        super().__init__()
        self.args = args
        nf = self.args.nf

        self.pooling = nn.MaxPool3d(kernel_size=2)

        self.conv1 = ConvBlock(in_c, nf)
        self.conv2 = ConvBlock(nf, nf*2)
        self.conv3 = ConvBlock(nf*2, nf*4)
        self.conv4 = ConvBlock(nf*4, nf*8)
        self.conv5 = ConvBlock(nf*8, nf*16)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pooling(c1)

        c2 = self.conv2(p1)
        p2 = self.pooling(c2)

        c3 = self.conv3(p2)
        p3 = self.pooling(c3)

        c4 = self.conv4(p3)
        p4 = self.pooling(c4)

        c5 = self.conv5(p4)

        enc_list = [c1, c2, c3, c4, c5]
        return enc_list


class ReconBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, pad=1):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.conv = ConvBlock(out_c, out_c, kernel=kernel, pad=pad)

    def forward(self, x, to_size):
        x = F.upsample(x, size=to_size, mode='trilinear')
        x = self.conv1x1(x)
        x = self.conv(x)

        return x


class FeatureRecon(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nf = self.args.nf

        self.recon1 = ReconBlock(nf*2, nf)
        self.recon2 = ReconBlock(nf*4, nf*2)
        self.recon3 = ReconBlock(nf*8, nf*4)
        self.recon4 = ReconBlock(nf*16, nf*8)

    def forward(self, enc_list):
        c1, c2, c3, c4, c5 = enc_list

        a1 = self.recon1(c2, c1.shape[2:])
        a2 = self.recon2(c3, c2.shape[2:])
        a3 = self.recon3(c4, c3.shape[2:])
        a4 = self.recon4(c5, c4.shape[2:])

        recon_list = [a1, a2, a3, a4]

        return recon_list


class TeacherDecoder(nn.Module):
    def __init__(self, args, out_c=1):
        super().__init__()
        self.args = args
        nf = self.args.nf

        self.conv = ConvBlock(nf*16, nf*16)
        self.up1 = UpMerge(nf*16, nf*8)
        self.up2 = UpMerge(nf*8, nf*4)
        self.up3 = UpMerge(nf*4, nf*2)
        self.up4 = UpMerge(nf*2, nf)

        self.out = nn.Sequential(
            nn.Conv3d(nf, out_c, kernel_size=1, stride=1, bias=False),
            # nn.Tanh()
            # nn.LeakyReLU()
        )

    def forward(self, enc_list):
        c1, c2, c3, c4, c5 = enc_list

        c5 = self.conv(c5)
        u1 = self.up1(c5, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)
        out = self.out(u4)

        return out


# class TeacherNet(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#
#         self.encoder = TeacherEncoder(args)
#         self.decoder = TeacherDecoder(args)
#
#     def forward(self, x):
#         enc_list = self.encoder(x)
#         out = self.decoder(enc_list)
#         return out


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

        self.lambda_img = self.args.lambda_img

    def training(self, device, fold, datetime_train):

        fold_name = 'Fold_%02d' % fold
        val_idx = [fold]

        ##### Directory
        dir_log = f'{self.path_log}/{datetime_train}_T'
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
        loss_L1 = nn.L1Loss()
        loss_MSE = nn.MSELoss()

        ##### Model
        encoder = nn.DataParallel(TeacherEncoder(self.args))
        decoder = nn.DataParallel(TeacherDecoder(self.args))
        featrecon = nn.DataParallel(FeatureRecon(self.args))

        encoder.apply(weights_init_normal)
        decoder.apply(weights_init_normal)
        featrecon.apply(weights_init_normal)

        encoder.to(device)
        decoder.to(device)
        featrecon.to(device)

        optimizer_E = torch.optim.Adam(encoder.parameters(), lr=self.lr, betas=(0.9, 0.999))
        optimizer_D = torch.optim.Adam(decoder.parameters(), lr=self.lr, betas=(0.9, 0.999))
        optimizer_F = torch.optim.Adam(featrecon.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # scheduler_E = torch.optim.lr_scheduler.StepLR(optimizer_E, step_size=1, gamma=0.99)
        # scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=1, gamma=0.99)
        # scheduler_F = torch.optim.lr_scheduler.StepLR(optimizer_F, step_size=1, gamma=0.99)

        logger.debug('============================================')
        logger.debug('[Teacher]')
        logger.debug(str(encoder))
        logger.debug(str(decoder))
        logger.debug(str(featrecon))
        logger.debug('Learning Rate: %.5f' % self.lr)
        logger.debug('Lambda for image-level reconstruction loss: %d' % self.lambda_img)
        logger.debug('============================================')

        ##### Training
        best = {'epoch': 0, 'psnr': 0, 'ssim': 0}
        patience = 0

        for epoch in tqdm(range(1, self.epochs + 1), desc='Epoch'):
            update = False
            train_loss = {'loss_T': 0, 'loss_FR': 0}
            encoder.train()
            decoder.train()
            featrecon.train()

            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Batch'):
                # logger.debug(f'[Epoch: {epoch} | Batch No.: {i}]')

                real_y = Variable(batch['y']).to(device)

                enc_list = encoder(real_y)

                frecon_list = featrecon(enc_list)
                loss_FR = loss_MSE(enc_list[0], frecon_list[0]) + loss_MSE(enc_list[1], frecon_list[1]) + loss_MSE(enc_list[2], frecon_list[2]) + loss_MSE(enc_list[3], frecon_list[3])

                #### Add noise
                for list_idx in range(len(enc_list)):
                    noised = enc_list[list_idx] + torch.rand(enc_list[list_idx].shape).to(device) * 1. + 0.
                    enc_list[list_idx] = noised

                pred_y = decoder(enc_list)
                loss_T = self.lambda_img * loss_L1(real_y, pred_y) + loss_FR

                optimizer_E.zero_grad()
                optimizer_D.zero_grad()
                optimizer_F.zero_grad()
                loss_T.backward()
                optimizer_E.step()
                optimizer_D.step()
                optimizer_F.step()

                train_loss['loss_T'] += loss_T.item()
                train_loss['loss_FR'] += loss_FR.item()

            if epoch % self.checkpoint_sample == 0:
                real_y_list, pred_y_list, _ = self.prediction(val_dataloader, encoder, decoder, device,
                                                              save_pred_path=f'{dir_result}/{epoch}_')
            else:
                real_y_list, pred_y_list, _ = self.prediction(val_dataloader, encoder, decoder, device)

            val_psnr, val_ssim = cal_psnr_ssim_list(real_y_list, pred_y_list)
            loss_T_ = train_loss['loss_T'] / len(train_dataloader)
            loss_FR_ = train_loss['loss_FR'] / len(train_dataloader)
            if best['psnr'] < val_psnr and best['ssim'] < val_ssim:
                torch.save(encoder.state_dict(), f'{dir_model}/teacher_encoder.pth')
                torch.save(decoder.state_dict(), f'{dir_model}/teacher_decoder.pth')
                torch.save(featrecon.state_dict(), f'{dir_model}/feature_recon.pth')
                best['epoch'] = epoch
                best['psnr'] = val_psnr
                best['ssim'] = val_ssim
                update = True
                patience = 0
            else:
                patience += 1

            logger.info(f'[Epoch: {epoch}/{self.epochs}]')
            # logger.info(f'[Epoch: {epoch}/{args.epochs}] lr: {scheduler_E.get_last_lr()}')
            logger.info(
                f'T loss: {round(loss_T_, 4)} | FR loss: {round(loss_FR_, 4)} | val_psnr: {round(val_psnr, 4)} | val_ssim: {round(val_ssim, 4)} | update: {str(update)}({patience})'
            )

            writer.add_scalar('loss T', loss_T_, epoch)
            writer.add_scalar('loss FR', loss_FR_, epoch)
            writer.add_scalar('val_psnr', val_psnr, epoch)
            writer.add_scalar('val_ssim', val_ssim, epoch)

            if patience == self.stop_patience:
                logger.info(
                    f'-------------------------------------------- Early Stopping ! Patience: {self.stop_patience}')
                break

            # scheduler_E.step()
            # scheduler_D.step()
            # scheduler_F.step()

        writer.close()

        logger.info('============================================')
        logger.info(f'[Best Performance of Validation]')
        logger.info(
            f'''Epoch: {best['epoch']} | PSNR: {best['psnr']} | SSIM: {best['ssim']}''')
        logger_closing(logger, stream_handler, file_handler)

        del encoder, decoder, featrecon

        ##### Load best model
        encoder_dict = torch.load(f'{dir_model}/teacher_encoder.pth')
        decoder_dict = torch.load(f'{dir_model}/teacher_decoder.pth')
        encoder = nn.DataParallel(TeacherEncoder(self.args)).to(device)
        decoder = nn.DataParallel(TeacherDecoder(self.args)).to(device)
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)

        # ##### Visualize feature maps
        # dir_fmap = f'{dir_log}/fmap/{fold_name}/'
        # os.makedirs(dir_fmap, exist_ok=True)
        # visualize_feature_maps_self(val_dataloader, encoder, device, dir_fmap)

        ##### Testing
        # self.testing(device, datetime_train, save_output=True)
        dir_all = f'{dir_log}/result_all/'
        os.makedirs(dir_all, exist_ok=True)
        logger_all, stream_handler_all, file_handler_all = logger_setting(file_name=f'{dir_all}/log_all.log')
        real_y_list, pred_y_list, patient_ids = self.prediction(val_dataloader, encoder, decoder, device,
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
        dir_log = f'{self.path_log}/{datetime_train}_T'
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
            dir_model = f'{dir_log}/model/{fold_name}'
            encoder_dict = torch.load(f'{dir_model}/teacher_encoder.pth')
            decoder_dict = torch.load(f'{dir_model}/teacher_decoder.pth')
            encoder = nn.DataParallel(TeacherEncoder(self.args)).to(device)
            decoder = nn.DataParallel(TeacherDecoder(self.args)).to(device)
            encoder.load_state_dict(encoder_dict)
            decoder.load_state_dict(decoder_dict)

            data_path = [f'{self.path_dataset}/S_{fold_name[-2:]}']
            dataset = MRI_7t(data_path)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            # real_y, pred_y, patient_ids = prediction(dataloader, generator_enc, generator_dec, device, dir_all)
            if save_output:
                real_y, pred_y, patient_ids = self.prediction(dataloader, encoder, decoder, device, dir_all)
            else:
                real_y, pred_y, patient_ids = self.prediction(dataloader, encoder, decoder, device)

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
        logger_closing(logger_all, stream_handler_all, file_handler_all)


    def prediction(self, dataloader, encoder, decoder, device, save_pred_path=False):
        patient_ids = []
        real_y_list = []
        pred_y_list = []

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for batch in dataloader:
                real_y = Variable(batch['y']).to(device)
                pred_y = decoder(encoder(real_y))

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
                        # save_nii(real_y, f'{save_pred_path}{patient_id}_real_y')
                        save_nii(pred_y_, f'{save_pred_path}{patient_id}_recon_y')

        return real_y_list, pred_y_list, patient_ids


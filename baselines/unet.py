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
from data import MRI_7t, IBSR, save_nii, segmap_to_onehot, fill_unlabeled
from metrics import cal_dice_score_list
from keeper import KnowledgeKeeperNet, FusionModules


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, pad=1, dropout=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class UpMerge(nn.Module): # Deconvolution
    def __init__(self, in_c, out_c, kernel=3, pad=1):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )
        self.conv = ConvBlock(out_c*2, out_c, kernel=kernel, pad=pad)

    def forward(self, x, skip_x):
        x = self.deconv(x)
        if x.shape[2:] != skip_x.shape[2:]:
            x = match_size(x, skip_x.shape[2:])
        x = torch.cat((x, skip_x), dim=1)
        x = self.conv(x)
        return x


class SegEncoder(nn.Module):
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


class SegDecoder(nn.Module):
    def __init__(self, args, out_c=4):
        super().__init__()
        self.args = args
        nf = self.args.nf

        self.conv = ConvBlock(nf*16, nf*16)
        self.up1 = UpMerge(nf*16, nf*8)
        self.up2 = UpMerge(nf*8, nf*4)
        self.up3 = UpMerge(nf*4, nf*2)
        self.up4 = UpMerge(nf*2, nf)

        self.out = nn.Sequential(
            nn.Conv3d(nf, out_c, kernel_size=1, stride=1, bias=False)
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


class Implementation(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.path_dataset_7T = self.args.path_dataset_7T
        self.path_dataset_IBSR = self.args.path_dataset_IBSR
        self.path_log = self.args.path_log
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.lr = self.args.lr
        self.checkpoint_sample = self.args.checkpoint_sample
        self.stop_patience = self.args.stop_patience

        self.base = self.args.base
        self.guided = self.args.guided
        self.train_net = self.args.train_net
        self.dataset = self.args.dataset

        self.tissue_num = self.args.tissue_num
        self.datetime_teacher = self.args.datetime_teacher
        self.datetime_keeper = self.args.datetime_keeper

    def training(self, device, fold, datetime_train):

        ##### Directory & Training Log
        if isinstance(fold, int):
            fold_name = 'Fold_%02d' % fold
        elif isinstance(fold, str):
            fold_name = 'Fold_' + fold

        if self.guided:
            dir_log = f'{self.path_log}/{self.datetime_teacher}_T_{self.datetime_keeper}_K_{datetime_train}_TissueSeg{self.train_net}_{self.dataset}_Guided{self.base}'
        else:
            assert self.train_net == 'ED', 'train_net should be ED(encoder and deocder)'
            dir_log = f'{self.path_log}/{datetime_train}_TissueSeg_{self.dataset}_{self.base}_Baseline'

        dir_model = f'{dir_log}/model/{fold_name}'
        dir_tboard = f'{dir_log}/tboard/{fold_name}'
        directory = [dir_log, dir_model, dir_tboard]
        for dir in directory:
            os.makedirs(dir, exist_ok=True)
        logger, stream_handler, file_handler = logger_setting(file_name=f'{dir_log}/log_{fold_name}.log')

        writer = SummaryWriter(dir_tboard)


        ##### Dataset Load
        if self.dataset == '7T':
            val_idx = [fold]
            train_data_path = []
            val_data_path = []
            for folder_name in sorted(os.listdir(self.path_dataset_7T)):
                _, patient_id = folder_name.split('_')  # folder_name example: S_01
                if int(patient_id) in val_idx:
                    val_data_path.append(f'{self.path_dataset_7T}/{folder_name}')
                else:
                    train_data_path.append(f'{self.path_dataset_7T}/{folder_name}')

            logger.info(f'''Valid data: {[path.split('/')[-1] for path in val_data_path]}''')
            logger.info(f'''Train data: {[path.split('/')[-1] for path in train_data_path]}''')

            train_dataset = MRI_7t(train_data_path, train=True, seg=True)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            val_dataset = MRI_7t(val_data_path, seg=True)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            test_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        elif 'IBSR' in self.dataset:
            val_idx = [1, 3, 18]
            train_idx = [2, 4, 5, 6, 7, 8, 9]
            test_idx = [10, 11, 12, 13, 14, 15, 16, 17]

            train_data_path = []
            val_data_path = []
            test_data_path = []
            for folder_name in sorted(os.listdir(self.path_dataset_IBSR)):
                _, patient_id = folder_name.split('_')  # folder_name example: IBSR_01
                if int(patient_id) in val_idx:
                    val_data_path.append(f'{self.path_dataset_IBSR}/{folder_name}')
                elif int(patient_id) in train_idx:
                    train_data_path.append(f'{self.path_dataset_IBSR}/{folder_name}')
                elif int(patient_id) in test_idx:
                    test_data_path.append(f'{self.path_dataset_IBSR}/{folder_name}')

            logger.info(f'''Train data: {[path.split('/')[-1] for path in train_data_path]}''')
            logger.info(f'''Valid data: {[path.split('/')[-1] for path in val_data_path]}''')
            logger.info(f'''Test data: {[path.split('/')[-1] for path in test_data_path]}''')

            train_dataset = IBSR(train_data_path, train=True, tissue_seg=True)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            val_dataset = IBSR(val_data_path, tissue_seg=True)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            test_dataset = IBSR(test_data_path, tissue_seg=True)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


        ##### Initialize
        loss_CCE = nn.CrossEntropyLoss()


        ##### Model
        if self.train_net == 'ED':
            encoder = nn.DataParallel(SegEncoder(self.args)).to(device)
            encoder.apply(weights_init_normal)
            optimizer_E = torch.optim.Adam(encoder.parameters(), lr=self.lr, betas=(0.9, 0.999))

        decoder = nn.DataParallel(SegDecoder(self.args)).to(device)
        decoder.apply(weights_init_normal)
        optimizer_D = torch.optim.Adam(decoder.parameters(), lr=self.lr, betas=(0.9, 0.999))

        if self.guided:
            fusion = nn.DataParallel(FusionModules(self.args)).to(device)
            fusion.apply(weights_init_normal)
            optimizer_F = torch.optim.Adam(fusion.parameters(), lr=self.lr, betas=(0.9, 0.999))


        logger.debug('============================================')
        ##### Pretrained model (Knowledge Keeper)
        if self.guided:
            pretrain_path = f'{self.path_log}/{self.datetime_teacher}_T_{self.datetime_keeper}_K/model/{fold_name}'
            keeper = nn.DataParallel(KnowledgeKeeperNet(self.args)).to(device)
            keeper.load_state_dict(torch.load(pretrain_path + '/knowledge_keeper.pth'))
            for param in keeper.parameters():
                param.requires_grad = False
            logger.info(f'Fold: {fold_name}')
            logger.debug(f'Pretrained model: {pretrain_path}')

        logger.debug('[Segmentation Model]')
        if self.train_net == 'ED':
            logger.debug(str(encoder))
        logger.debug(str(decoder))
        logger.debug('Learning Rate: %.5f' % self.lr)
        logger.debug('============================================')

        ##### Training
        best = {'epoch': 0, 'score': 0, 'loss': np.inf}
        patience = 0

        for epoch in tqdm(range(1, self.epochs + 1), desc='Epoch'):
            update = False
            train_loss = {'loss_total': 0}
            if self.train_net == 'ED':
                encoder.train()
            decoder.train()
            if self.guided:
                fusion.train()

            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Batch'):
                real_x = Variable(batch['x']).to(device)
                real_y = Variable(batch['y']).to(device)
                if self.guided:
                    if self.train_net == 'ED':
                        g1, g2, g3, g4, g5 = keeper(real_x)
                        e1, e2, e3, e4, e5 = encoder(real_x)
                        enc_list = fusion([g1, g2, g3, g4, g5], [e1, e2, e3, e4, e5])
                        pred_y = decoder(enc_list)
                    else:
                        pred_y = decoder(keeper(real_x))
                else:
                    pred_y = decoder(encoder(real_x))

                loss_total = loss_CCE(pred_y, real_y)

                if self.train_net == 'ED':
                    optimizer_E.zero_grad()
                optimizer_D.zero_grad()
                if self.guided:
                    optimizer_F.zero_grad()
                loss_total.backward()
                if self.train_net == 'ED':
                    optimizer_E.step()
                optimizer_D.step()
                if self.guided:
                    optimizer_F.step()

                train_loss['loss_total'] += loss_total.item()

            if self.guided:
                if self.train_net == 'ED':
                    real_y_list, pred_y_list, _ = self.prediction_guided_ED(val_dataloader, encoder, decoder, keeper, fusion, device)
                else:
                    real_y_list, pred_y_list, _ = self.prediction(val_dataloader, keeper, decoder, device)
            else:
                real_y_list, pred_y_list, _ = self.prediction(val_dataloader, encoder, decoder, device)

            val_scores_list = cal_dice_score_list(real_y_list, pred_y_list)
            val_score = 0
            for score in val_scores_list:
                val_score = val_score + score

            loss_total_ = train_loss['loss_total'] / len(train_dataloader)

            if best['score'] < val_score:
                if self.train_net == 'ED':
                    torch.save(encoder.state_dict(), f'{dir_model}/seg_encoder.pth')
                torch.save(decoder.state_dict(), f'{dir_model}/seg_decoder.pth')
                if self.guided:
                    torch.save(fusion.state_dict(), f'{dir_model}/fusion.pth')
                best['epoch'] = epoch
                best['score'] = val_score
                best['loss'] = loss_total_
                update = True
                patience = 0
            else:
                patience += 1

            logger.info(f'[Epoch: {epoch}/{self.epochs}]')
            logger.info(
                f'Total loss: {round(loss_total_, 4)} | val_score_total: {round(val_score, 4)} | val_score_bg: {round(val_scores_list[0], 4)} | val_score_CSF: {round(val_scores_list[1], 4)} | val_score_GM: {round(val_scores_list[2], 4)} | val_score_WM: {round(val_scores_list[3], 4)} | valid_update: {str(update)}({patience})'
            )

            writer.add_scalar('loss', loss_total_, epoch)

            if patience == self.stop_patience:
                logger.info(
                    f'-------------------------------------------- Early Stopping ! Patience: {self.stop_patience}')
                break

        writer.close()

        logger.info('============================================')
        logger.info(f'[Best Performance for Validation]')
        logger.info(f'''Epoch: {best['epoch']} | Score: {best['score']}''')
        logger_closing(logger, stream_handler, file_handler)

        ##### Load best model
        if self.train_net == 'ED':
            encoder_dict = torch.load(f'{dir_model}/seg_encoder.pth')
            encoder = nn.DataParallel(SegEncoder(self.args)).to(device)
            encoder.load_state_dict(encoder_dict)
        decoder_dict = torch.load(f'{dir_model}/seg_decoder.pth')
        decoder = nn.DataParallel(SegDecoder(self.args)).to(device)
        decoder.load_state_dict(decoder_dict)
        if self.guided:
            fusion_dict = torch.load(f'{dir_model}/fusion.pth')
            fusion = nn.DataParallel(FusionModules(self.args)).to(device)
            fusion.load_state_dict(fusion_dict)

        ##### Testing
        dir_all = f'{dir_log}/result_all/'
        os.makedirs(dir_all, exist_ok=True)
        logger_all, stream_handler_all, file_handler_all = logger_setting(file_name=f'{dir_all}/log_all.log')
        if self.guided:
            if self.train_net == 'ED':
                real_y_list, pred_y_list, patient_ids = self.prediction_guided_ED(test_dataloader, encoder, decoder, keeper, fusion, device, save_pred_path=f'{dir_all}/{fold_name}_')
            else:
                real_y_list, pred_y_list, patient_ids = self.prediction(test_dataloader, keeper, decoder, device, save_pred_path=f'{dir_all}/{fold_name}_')
        else:
            real_y_list, pred_y_list, patient_ids = self.prediction(test_dataloader, encoder, decoder, device, save_pred_path=f'{dir_all}/{fold_name}_')
        scores_mean, scores = cal_dice_score_list(real_y_list, pred_y_list, return_total=True)
        mean_bg, mean_CSF, mean_GM, mean_WM = scores_mean
        scores_bg, scores_CSF, scores_GM, scores_WM = scores

        for idx, patient_id in enumerate(patient_ids):
            score_bg = scores_bg[idx]
            score_CSF = scores_CSF[idx]
            score_GM = scores_GM[idx]
            score_WM = scores_WM[idx]
            logger_all.info(f'{fold_name} | {patient_id} | {score_bg} | {score_CSF} | {score_GM} | {score_WM}')
        logger_closing(logger_all, stream_handler_all, file_handler_all)

        logger, stream_handler, file_handler = logger_setting(file_name=f'{dir_log}/log_{fold_name}.log')
        logger.info('[Test Result]')
        logger.info(f'Background: {round(mean_bg,4)} | CSF: {round(mean_CSF,4)} | GM: {round(mean_GM,4)} | WM: {round(mean_WM,4)}')
        logger_closing(logger, stream_handler, file_handler)

        torch.cuda.empty_cache()


    def testing(self, device, datetime_train, save_output=True):
        if self.guided:
            dir_log = f'{self.path_log}/{self.datetime_teacher}_T_{self.datetime_keeper}_K_{datetime_train}_TissueSeg{self.train_net}_{self.dataset}_Guided{self.base}'
        else:
            assert self.train_net == 'ED', 'train_net should be ED(encoder and deocder)'
            dir_log = f'{self.path_log}/{datetime_train}_TissueSeg_{self.dataset}_{self.base}_Baseline'
        dir_all = f'{dir_log}/result_all/'
        os.makedirs(dir_all, exist_ok=True)

        logger_all, stream_handler_all, file_handler_all = logger_setting(file_name=f'{dir_all}/log_all.log')
        logger_all.info(f'Model: {datetime_train}')

        fold_mean_bg = []
        fold_mean_CSF = []
        fold_mean_GM = []
        fold_mean_WM = []

        logger_all.info('[Patient ID | Background | CSF | GM | WM]')

        fold_names = sorted(os.listdir(f'{dir_log}/model'))

        for fold_name in fold_names:
            dir_model = f'{dir_log}/model/{fold_name}'
            if self.train_net == 'ED':
                encoder_dict = torch.load(f'{dir_model}/seg_encoder.pth')
                encoder = nn.DataParallel(SegEncoder(self.args)).to(device)
                encoder.load_state_dict(encoder_dict)
            decoder_dict = torch.load(f'{dir_model}/seg_decoder.pth')
            decoder = nn.DataParallel(SegDecoder(self.args)).to(device)
            decoder.load_state_dict(decoder_dict)
            if self.guided:
                dir_pretrain = f'{self.path_log}/{self.datetime_teacher}_T_{self.datetime_keeper}_K/model/{fold_name}'
                keeper_dict = torch.load(f'{dir_pretrain}/knowledge_keeper.pth')
                keeper = nn.DataParallel(KnowledgeKeeperNet(self.args)).to(device)
                keeper.load_state_dict(keeper_dict)

                fusion_dict = torch.load(f'{dir_model}/fusion.pth')
                fusion = nn.DataParallel(FusionModules(self.args)).to(device)
                fusion.load_state_dict(fusion_dict)

            if dataset == '7T':
                data_path = [f'{self.path_dataset_7T}/S_{fold_name[-2:]}']
                dataset = MRI_7t(data_path, seg=True)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            elif 'IBSR' in dataset:
                test_idx = [10, 11, 12, 13, 14, 15, 16, 17]
                data_path = []
                for folder_name in sorted(os.listdir(self.path_dataset_IBSR)):
                    _, patient_id = folder_name.split('_')  # folder_name example: IBSR_01
                    if int(patient_id) in test_idx:
                        data_path.append(f'{self.path_dataset_IBSR}/{folder_name}')
                dataset = IBSR(data_path, tissue_seg=True)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            if save_output:
                if self.guided:
                    if self.train_net == 'ED':
                        real_y_list, pred_y_list, patient_ids = self.prediction_guided_ED(dataloader, encoder, decoder, keeper, fusion, device, save_pred_path=f'{dir_all}/{fold_name}_')
                    else:
                        real_y_list, pred_y_list, patient_ids = self.prediction(dataloader, keeper, decoder, device, save_pred_path=f'{dir_all}/{fold_name}_')
                else:
                    real_y_list, pred_y_list, patient_ids = self.prediction(dataloader, encoder, decoder, device, save_pred_path=f'{dir_all}/{fold_name}_')
            else:
                if self.guided:
                    if self.train_net == 'ED':
                        real_y_list, pred_y_list, patient_ids = self.prediction_guided_ED(dataloader, encoder, decoder, keeper, fusion, device)
                    else:
                        real_y_list, pred_y_list, patient_ids = self.prediction(dataloader, keeper, decoder, device)
                else:
                    real_y_list, pred_y_list, patient_ids = self.prediction(dataloader, encoder, decoder, device)

            scores_mean, scores = cal_dice_score_list(real_y_list, pred_y_list, return_total=True)
            mean_bg, mean_CSF, mean_GM, mean_WM = scores_mean
            scores_bg, scores_CSF, scores_GM, scores_WM = scores

            for idx, patient_id in enumerate(patient_ids):
                score_bg = scores_bg[idx]
                score_CSF = scores_CSF[idx]
                score_GM = scores_GM[idx]
                score_WM = scores_WM[idx]
                logger_all.info(f'{fold_name} | {patient_id} | {score_bg} | {score_CSF} | {score_GM} | {score_WM}')

            fold_mean_bg.append(mean_bg)
            fold_mean_CSF.append(mean_CSF)
            fold_mean_GM.append(mean_GM)
            fold_mean_WM.append(mean_WM)

            logger_all.info('=====================================================')

        fold_mean_bg = np.array(fold_mean_bg)
        fold_mean_CSF = np.array(fold_mean_CSF)
        fold_mean_GM = np.array(fold_mean_GM)
        fold_mean_WM = np.array(fold_mean_WM)

        np.save(f'{dir_all}all_fold_mean_bg', fold_mean_bg)
        np.save(f'{dir_all}all_fold_mean_CSF', fold_mean_CSF)
        np.save(f'{dir_all}all_fold_mean_GM', fold_mean_GM)
        np.save(f'{dir_all}all_fold_mean_WM', fold_mean_WM)

        logger_all.info('[Test Result]')
        logger_all.info(f'Background: {round(fold_mean_bg.mean(), 4)} | CSF: {round(fold_mean_CSF.mean(), 4)} | GM: {round(fold_mean_GM.mean(), 4)} | WM: {round(fold_mean_WM.mean(), 4)}')
        logger_closing(logger_all, stream_handler_all, file_handler_all)


    def prediction(self, dataloader, encoder, decoder, device, save_pred_path=False):
        patient_ids = []
        real_y_list = []
        pred_y_list = []

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for batch in dataloader:
                real_x = Variable(batch['x']).to(device)
                real_y = Variable(batch['y']).to(device)

                pred_y = decoder(encoder(real_x))
                pred_y = F.softmax(pred_y, dim=1)

                pred_y = torch.argmax(pred_y, dim=1, keepdim=False)

                real_y = real_y.cpu().detach().numpy()
                pred_y = pred_y.cpu().detach().numpy()
                if self.dataset == 'IBSR':
                    ulb = Variable(batch['ulb']).to(device)
                    ulb = ulb.cpu().detach().numpy()

                for idx in range(pred_y.shape[0]):
                    patient_id = str(batch['patient_id'][idx])
                    patient_ids.append(patient_id)

                    real_y_list.append(real_y[idx])

                    ### Segmentation
                    pred_y_ = pred_y[idx]
                    pred_y_ = segmap_to_onehot(pred_y_.squeeze())
                    if self.dataset == 'IBSR':
                        pred_y_ = fill_unlabeled(ulb[idx], pred_y_)

                    pred_y_list.append(pred_y_)

                    if save_pred_path:
                        save_nii(np.argmax(pred_y_, axis=0).astype(float), f'{save_pred_path}{patient_id}_pred_y')

        return real_y_list, pred_y_list, patient_ids


    def prediction_guided_ED(self, dataloader, encoder, decoder, keeper, fusion, device, save_pred_path=False):
        patient_ids = []
        real_y_list = []
        pred_y_list = []

        encoder.eval()
        decoder.eval()
        fusion.eval()
        with torch.no_grad():
            for batch in dataloader:
                real_x = Variable(batch['x']).to(device)
                real_y = Variable(batch['y']).to(device)

                g1, g2, g3, g4, g5 = keeper(real_x)
                e1, e2, e3, e4, e5 = encoder(real_x)
                enc_list = fusion([g1, g2, g3, g4, g5], [e1, e2, e3, e4, e5])
                pred_y = decoder(enc_list)

                pred_y = F.softmax(pred_y, dim=1)

                pred_y = torch.argmax(pred_y, dim=1, keepdim=False)

                real_y = real_y.cpu().detach().numpy()
                pred_y = pred_y.cpu().detach().numpy()
                if self.dataset == 'IBSR':
                    ulb = Variable(batch['ulb']).to(device)
                    ulb = ulb.cpu().detach().numpy()

                for idx in range(pred_y.shape[0]):
                    patient_id = str(batch['patient_id'][idx])
                    patient_ids.append(patient_id)

                    real_y_list.append(real_y[idx])

                    ### Segmentation
                    pred_y_ = pred_y[idx]
                    pred_y_ = segmap_to_onehot(pred_y_.squeeze())
                    if self.dataset == 'IBSR':
                        pred_y_ = fill_unlabeled(ulb[idx], pred_y_)

                    pred_y_list.append(pred_y_)

                    if save_pred_path:
                        save_nii(np.argmax(pred_y_, axis=0).astype(float), f'{save_pred_path}{patient_id}_pred_y')

        return real_y_list, pred_y_list, patient_ids

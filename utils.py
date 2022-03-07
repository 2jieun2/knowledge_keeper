import logging
import torch
from torch import nn


def logger_setting(file_name):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=file_name)

    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # stream_handler.setFormatter(formatter)
    # file_handler.setFormatter(formatter)

    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger, stream_handler, file_handler


def logger_closing(logger, stream_handler, file_handler):
    stream_handler.close()
    logger.removeHandler(stream_handler)
    file_handler.close()
    logger.removeHandler(file_handler)
    del logger, stream_handler, file_handler


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        # nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity="leaky_relu")
    elif classname.find('ConvTranspose3d') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity="leaky_relu")
    elif classname.find('BatchNorm3d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def match_size(x, size):
    _, _, h1, w1, d1 = x.shape
    h2, w2, d2 = size

    while d1 != d2:
        if d1 < d2:
            x = nn.functional.pad(x, (0, 1), mode='constant', value=0)
            d1 += 1
        else:
            x = x[:, :, :, :, :d2]
            break
    while w1 != w2:
        if w1 < w2:
            x = nn.functional.pad(x, (0, 0, 0, 1), mode='constant', value=0)
            w1 += 1
        else:
            x = x[:, :, :, :w2, :]
            break
    while h1 != h2:
        if h1 < h2:
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, 1), mode='constant', value=0)
            h1 += 1
        else:
            x = x[:, :, :h2, :, :]
            break
    return x


def loss_dist_match(real, fake):
    loss = 0
    loss_MSE = nn.MSELoss()
    for b in range(real.shape[0]):
        real_vol = real[b,:,:,:,:] # (c, h, w, d)
        fake_vol = fake[b,:,:,:,:]

        real_std, real_mu = torch.std_mean(real_vol, dim=0, unbiased=False)
        fake_std, fake_mu = torch.std_mean(fake_vol, dim=0, unbiased=False)

        loss += loss_MSE(real_std, fake_std) + loss_MSE(real_mu, fake_mu)
    return loss
import os
import argparse
import GPUtil
from datetime import datetime
import torch


########## Argument
parser = argparse.ArgumentParser()

##### Setting
parser.add_argument('--net', type=str, default='T') # T(teacher) / K(keeper) / S(segmentation)
parser.add_argument('--mode', type=str, default='train') # train / test
parser.add_argument('--gpu', type=int, default=-1)
### Segmentation
parser.add_argument('--base', type=str, default='UNet') # baseline models: UNet / USegNet / VoxResNet / RPNet
parser.add_argument('--guided', type=int, default=0) # guided by keeper (1, True) or not (0, False)
parser.add_argument('--train_net', type=str, default='ED') # train a segmentation encoder-decoder (ED) / only decoder (else)
parser.add_argument('--dataset', type=str, default='IBSR') # dataset for segmentation: IBSR / 7T

##### Dataset path & Directory for saving a log file
parser.add_argument('--path_dataset_7T', type=str, default='/pared_3T_7T')
parser.add_argument('--path_dataset_IBSR', type=str, default='/IBSR')
parser.add_argument('--path_log', type=str, default='/log_KnowledgeKeeper')

##### Datetime(used for directory names) of a trained model
parser.add_argument('--datetime_teacher', type=str, default='yyyymmdd_hhmmss_T')
parser.add_argument('--datetime_keeper', type=str, default='yyyymmdd_hhmmss_K')
parser.add_argument('--datetime_seg', type=str, default='yyyymmdd_hhmmss_S')

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--nf', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--lambda_img', type=int, default=100)
parser.add_argument('--lambda_vox', type=int, default=100)
parser.add_argument('--lambda_adv', type=int, default=0.5)
parser.add_argument('--checkpoint_sample', type=int, default=20)
parser.add_argument('--stop_patience', type=int, default=50)

parser.add_argument('--tissue_num', type=int, default=4) # the number of tissue labels

args = parser.parse_args()

if __name__ == "__main__":
    torch.cuda.empty_cache()

    ########## GPU Configuration
    gpu_id = args.gpu
    # gpu_id = -1
    if gpu_id == -1:
        devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
    else:
        devices = "%d" % gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########## Folds
    if (args.net == 'SD' or args.net == 'S') and args.dataset == 'IBSR' and args.guided == False:
        folds = ['NA']
    else:
        folds = list(range(1, 16))

    ########## Implementation
    if args.net == 'T':
        from teacher import Implementation as Model
        datetime_train = args.datetime_teacher # for testing
    elif args.net == 'K':
        from keeper import Implementation as Model
        datetime_train = args.datetime_keeper # for testing
    elif args.net == 'SD' or args.net == 'S':
        if args.base == 'UNet':
            from baselines.unet import Implementation as Model
        elif args.base == 'VoxResNet':
            from baselines.voxresnet import Implementation as Model
        elif args.base == 'USegNet':
            from baselines.usegnet import Implementation as Model
        elif args.base == 'RPNet':
            from baselines.rpnet import Implementation as Model
        datetime_train = args.datetime_seg  # for testing

    model = Model(args)

    if args.mode == 'train':
        datetime_train = datetime.today().strftime('%Y%m%d_%H%M%S')
        for fold in folds:
            model.training(device, fold, datetime_train)
    elif args.mode == 'test':
        model.testing(device, datetime_train, save_output=False)

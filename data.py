import os
from glob import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from skimage.transform import resize, rescale
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage import rotate
from skimage import exposure


def save_nii(arr, path, affine=np.eye(4)):
    nii_img = nib.Nifti1Image(arr, affine=affine)
    nib.save(nii_img, path)


def load_nii(path_file):
    proxy = nib.load(path_file)
    array = proxy.get_fdata()
    return array


class MRI_7t(Dataset):
    def __init__(self, path_dataset, train=False, seg=False):
        self.patient_ids, self.x_list, self.y_list = self.get_dataset(path_dataset, seg)
        self.train = train
        self.seg = seg

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        x = self.x_list[index]
        y = self.y_list[index]

        x_h, x_w, x_d = x.shape
        y_h, y_w, y_d = y.shape

        if self.train:
            aug_index = np.random.rand(3)
            if aug_index[0] > 0.5:
                x, y = rand_rotate_specific(x, y)
            if aug_index[1] > 0.5:
                x, y = rand_scale(x, y, 0.8, 1.2)
            if aug_index[2] > 0.5:
                x, y = flip_by_axis(x, y, axis=-1)

        x = torch.from_numpy(x.copy()).float().view(1, x_h, x_w, x_d)
        if self.seg:
            y = segmap_to_onehot(y) # output shape: (num_labels, h, w, d)
            y = torch.from_numpy(y.copy()).float()
        else:
            y = torch.from_numpy(y.copy()).float().view(1, y_h, y_w, y_d)

        return {'patient_id': patient_id, 'x': x, 'y': y}

    def get_dataset(self, path_dataset, seg=False):
        patient_ids = []
        x_list = []
        y_list = []

        for path_data in path_dataset:
            patient_id = path_data.split('/')[-1]

            path_3t = f'{path_data}/3t_02_norm_crop.nii'
            if seg:
                path_7t = f'{path_data}/7t_02_norm_crop_tisseg.nii'
            else:
                path_7t = f'{path_data}/7t_02_norm_crop.nii'

            x = nib.load(path_3t).get_data()
            y = nib.load(path_7t).get_data()

            x_list.append(x)
            y_list.append(y)
            patient_ids.append(patient_id)

        return patient_ids, x_list, y_list


def segmap_to_onehot(y, num_labels=4):
    out = np.zeros((num_labels, y.shape[0], y.shape[1], y.shape[2]))
    for idx, label in enumerate(range(num_labels)):
        out[idx, :] = np.where(y == label, 1, 0)
    return out


def flip_by_axis(x, y, axis):
    x = np.flip(x, axis=axis)
    y = np.flip(y, axis=axis)
    return x, y


def rand_scale(x, y, range_min, range_max):
    scale_uniform = np.random.rand(1)[0]
    # scale_uniform = np.random.rand(3)
    scale = (range_max - range_min) * scale_uniform + range_min

    aff_matrix = np.array([[scale, 0, 0],
                           [0, scale, 0],
                           [0, 0, scale]])
    # aff_matrix = np.array([[scale[0], 0, 0],
    #                        [0, scale[1], 0],
    #                        [0, 0, scale[2]]])

    center = 0.5 * np.array(x.shape)
    offset = center - center.dot(aff_matrix)

    x = affine_transform(x, aff_matrix, offset=offset)
    y = affine_transform(y, aff_matrix, offset=offset)

    return x, y


def rand_rotate_specific(x, y, specific_angle=[90, 180, 270]):
    angle = np.random.choice(specific_angle)
    x = rotate(x, angle, reshape=False)
    y = rotate(y, angle, reshape=False)
    return x, y


def min_max_norm(data):
    return (data-np.min(data)) / (np.max(data)-np.min(data))


def clahe(arr, clip_limit=0.2):
    kernel_size = (arr.shape[0] // 5,
               arr.shape[1] // 5,
               arr.shape[2] // 5)
    kernel_size = np.array(kernel_size)
    out = [exposure.equalize_adapthist(im,
                                 kernel_size=kernel_size,
                                 clip_limit=clip_limit)
             for im in [arr]]
    return out[0]


def fill_unlabeled(mask, target): # mask -> unlabeled: 0, remaining: 1
    ### Fill the unlabeld regions with CSF values. https://www.nitrc.org/forum/message.php?msg_id=25702
    mul = np.tile(mask.squeeze(), (4, 1, 1, 1))
    mul[1] = 1
    out = np.multiply(target, mul)
    mask = np.where(mask == 1, 0, 1)  # unlabeled: 1, remaining: 0
    add = np.zeros_like(out)
    add[1] = mask.squeeze()
    out += add
    out = np.where(out == 2, 1, out)
    return out


# https://www.nitrc.org/projects/ibsr
class IBSR(Dataset):
    def __init__(self, path_dataset, train=False, tissue_seg=False):
        self.patient_ids, self.x_list, self.y_list, self.ulb_list = self.get_dataset_IBSR(path_dataset, tissue_seg)
        self.seg = tissue_seg
        self.train = train

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        x = self.x_list[index]
        y = self.y_list[index]

        x = clahe(x)

        if self.train:
            aug_index = np.random.rand(3)
            if aug_index[0] > 0.5:
                x, y = rand_rotate_specific(x, y)
            if aug_index[1] > 0.5:
                x, y = rand_scale(x, y, 0.8, 1.2)
            if aug_index[2] > 0.5:
                x, y = flip_by_axis(x, y, axis=-1)

        x_h, x_w, x_d = x.shape

        x = torch.from_numpy(x.copy()).float().view(1, x_h, x_w, x_d)

        y = segmap_to_onehot(y) # output shape: (num_labels, h, w, d)
        y = torch.from_numpy(y.copy()).float()

        if self.train == False:
            ulb = self.ulb_list[index]
            ulb = torch.from_numpy(ulb.copy()).float().view(1, x_h, x_w, x_d)
            return {'patient_id': patient_id, 'x': x, 'y': y, 'ulb': ulb}
        else:
            return {'patient_id': patient_id, 'x': x, 'y': y}

    def get_dataset_IBSR(self, path_dataset, tissue_seg=False):
        patient_ids = []
        x_list = []
        y_list = []
        ulb_list = []

        for path_data in path_dataset:
            patient_id = path_data.split('/')[-1].split('_')[-1]

            path_x = f'{path_data}/IBSR_{patient_id}_ana_strip_norm.nii'
            if tissue_seg: # The 'fill' files have any regions of zeros that are inside the brain mask set to 1 (the CSF value). https://www.nitrc.org/forum/message.php?msg_id=25702
                path_y = f'{path_data}/IBSR_{patient_id}_segTRI_fill_ana.nii'
            # else: # region_seg
            #     path_y = f'{path_data}/IBSR_{patient_id}_seg_ana.nii.gz'
            path_ulb = f'{path_data}/IBSR_{patient_id}_unlabeled.nii'

            x = nib.load(path_x).get_fdata().squeeze()
            y = nib.load(path_y).get_fdata().squeeze()
            ulb = nib.load(path_ulb).get_fdata().squeeze()

            x_list.append(x)
            y_list.append(y)
            patient_ids.append(patient_id)
            ulb_list.append(ulb)

        return patient_ids, x_list, y_list, ulb_list

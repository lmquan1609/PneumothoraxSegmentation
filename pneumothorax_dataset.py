import os

import numpy as np
import cv2
import pandas as pd

import torch
from albumentations.pytorch.transforms import ToTensor

class PneumothoraxDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, mode, transform=None, folder_index=None, folds_distr_path=None):
        self.transform = transform
        self.mode = mode

        self.train_image_path = os.path.join(data_folder, 'train')
        self.train_mask_path = os.path.join(data_folder, 'mask')
        self.test_image_path = os.path.join(data_folder, 'test')

        self.fold_index = None
        self.folds_distr_path = folds_distr_path
        self.set_mode(mode, folder_index)
        self.to_tensor = ToTensor()

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index

        if self.mode == 'train':
            folds = pd.read_csv(self.folds_distr_path)
            folds['fold'] = folds['fold'].astype(str)
            folds = folds[folds['fold'] != fold_index]

            self.train_list = folds['fname'].values.tolist()
            self.exist_labels = folds['exist_labels'].values.tolist()

            self.num_data = len(self.train_list)

        elif self.mode == 'val':
            folds = pd.read_csv(self.folds_distr_path)
            folds['fold'] = folds['fold'].astype(str)
            folds = folds[folds['fold'] == fold_index]

            self.val_list = folds['fname'].values.tolist()
            self.num_data = len(self.train_list)

        elif self.mode == 'test':
            self.test_list = sorted(os.listdir(self.test_image_path))
            self.num_data = len(self.test_list)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = cv2.imread(os.path.join(self.train_image_path, self.train_list[index]))
            if self.exist_labels[index] == 0:
                label = np.zeros((1024, 1024), dtype='uint8')
            else:
                label = cv2.imread(os.path.join(self.train_mask_path, self.train_list[index]), 0)
        elif self.mode == 'val':
            image = cv2.imread(os.path.join(self.train_image_path, self.val_list[index]))
            label = cv2.imread(os.path.join(self.train_mask_path, self.val_list[index]), 0)

        elif self.mode == 'test':
            image = cv2.imread(os.path.join(self.test_image_path, self.test_list[index]))

            if self.transform:
                sample = {'image':image}
                sample = self.transform(**sample)
                sample = self.to_tensor(**sample)
                image = sample['image']
            image_id = self.test_image_path[index].replace('.png', '')
            return image_id, image
        
        if self.transform:
            sample = {'image':image, 'mask':label}
            sample = self.transform(**sample)
            sample = self.to_tensor(**sample)
            image, label = sample['image'], sample['mask']

        return image, label

    def __len__(self):
        return self.num_data

class PneumoSampler(torch.utils.data.Sampler):
    def __init__(self, folds_distr_path, fold_index, demand_non_empty_prob):
        assert demand_non_empty_prob > 0, 'frequency of non-empty images must be greater than zero'
        self.fold_index = fold_index
        self.positive_prob = demand_non_empty_prob

        self.folds = pd.read_csv(folds_distr_path)
        self.folds['fold'] = self.folds['fold'].astype(str)
        self.folds = self.folds[self.folds['fold'] != fold_index].reset_index(drop=True)

        self.positive_indices = self.folds[self.folds['exist_labels'] == 1].index.values
        self.negative_indices = self.folds[self.folds['exist_labels'] == 0].index.values

        self.n_positive = self.positive_indices.shape[0]
        self.n_negative = int(self.n_positive * (1 - self.positive_prob) / self.positive_prob)

    def __iter__(self):
        negative_sample = np.random.choice(self.negative_indices, self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_indices)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative
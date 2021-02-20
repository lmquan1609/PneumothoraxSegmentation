import argparse
import os

import albumentations as albu
import torch

import cv2
from pathlib import Path

from models.ternausnet import AlbuNet
from albumentations.pytorch.transforms import ToTensor

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, help='Path to image')
    parser.add_argument('transforms', type=str, default='transforms/val_transforms_1024_old.json', help='Path to validation transform')
    parser.add_argument('--device', type=str, default='cpu', help='Whether to use GPU (cuda/cpu)')
    parser.add_argument('--checkpoints-path', type=str, default='experiments/albunet_valid/checkpoints', help='Path to checkpoint directory')
    parser.add_argument('--pipeline-name', type=str, default='albunet_1024', help='Name of experiment')
    parser.add_argument('--usefolds', type=int, default=5, help='Number of folds out of original 5 folds')
    return vars(parser.parse_args())

def build_checkpoints_list(cfg):
    usefolds = list(range(cfg['usefolds']))
    checkpoints_list = []
    for fold_id in usefolds:
        filename = f"{cfg['pipeline_name']}_fold_{fold_id}.pth"
        checkpoints_list.append(Path(cfg['checkpoints_path'], filename))
    
    return checkpoints_list

def predict(cfg):
    assert Path(cfg['image_path']).is_file(), 'image path is invalid'
    assert Path(cfg['transforms']).is_file(), 'Validation transform is invalid'

    image = cv2.imread(cfg['image_path'])
    model = AlbuNet(pretrained=False).to(cfg['device'])
    transform = albu.load(cfg['transforms'])

    to_tensor = ToTensor()

    checkpoints_list = build_checkpoints_list(cfg)
    mask = 0
    for pred_idx, checkpoint_path in enumerate(checkpoints_list):
        sample = transform(image=image)
        sample = to_tensor(**sample)
        image = sample['image'].unsqueeze(0).to(cfg['device'])

        print(checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        preds = model(image)
        curr_masks = torch.sigmoid(preds)
        curr_masks = curr_masks.squeeze(1).cpu().detach().numpy()
        mask = (mask * pred_idx + curr_masks) / (pred_idx + 1)
    return mask.squeeze(0)


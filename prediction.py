import argparse
import os

import albumentations as albu
import torch

import cv2
import numpy as np
import pydicom
from pathlib import Path
from imutils import paths

from models.ternausnet import AlbuNet
from albumentations.pytorch.transforms import ToTensor
from skimage.transform import resize

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dcm-path', type=str, help='Path to image')
    parser.add_argument('--transforms', type=str, default='transforms/val_transforms_1024_old.json', help='Path to validation transform')
    parser.add_argument('--device', type=str, default='cpu', help='Whether to use GPU (cuda/cpu)')
    parser.add_argument('--checkpoints-path', type=str, default='experiments/albunet_valid/checkpoints', help='Path to checkpoint directory')
    parser.add_argument('--pipeline-name', type=str, default='albunet_1024', help='Name of experiment')
    parser.add_argument('--usefolds', type=int, default=5, help='Number of folds out of original 5 folds')
    parser.add_argument('--image-size', type=int, default=1024, help='Image size')
    return vars(parser.parse_args())

def build_checkpoints_list(cfg):
    usefolds = list(range(cfg['usefolds']))
    checkpoints_list = []
    for fold_id in usefolds:
        filename = f"{cfg['pipeline_name']}_fold_{fold_id}.pth"
        checkpoints_list.append(Path(cfg['checkpoints_path'], filename))
    
    return checkpoints_list

def predict(cfg):
    # assert Path(cfg['dcm_path']).is_file() and cfg['dcm_path'][-3:] == 'dcm', 'image path is invalid'
    assert Path(cfg['transforms']).is_file(), 'Validation transform is invalid'

    model = AlbuNet(pretrained=False).to(cfg['device'])
    transform = albu.load(cfg['transforms'])
    to_tensor = ToTensor()

    for dcm_path in paths.list_files(cfg['dcm_path']):

        image = pydicom.read_file(dcm_path).pixel_array
        image = resize(image, (cfg['image_size'], cfg['image_size']))
        image = (image * 255).astype('uint8')
        image = np.dstack([image] * 3)
        
        # fn = cfg['dcm_path'][:cfg['dcm_path'].rfind('.')]
        # cv2.imwrite(fn + '.png', image)
        # print(f'DCM file is trasformed to PNG in {fn}.png')

        
        sample = transform(image=image)
        sample = to_tensor(**sample)
        image = sample['image'].unsqueeze(0).to(cfg['device'])

        checkpoints_list = build_checkpoints_list(cfg)
        mask = 0
        for pred_idx, checkpoint_path in enumerate(checkpoints_list):
            print(checkpoint_path)
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(cfg['device'])))
            model.eval()

            preds = model(image)
            curr_masks = torch.sigmoid(preds)
            curr_masks = curr_masks.squeeze(1).cpu().detach().numpy()
            print(f"tmp/{dcm_path[dcm_path.rfind(os.path.sep) + 1:dcm_path.rfind('.')]}_segmented.png")
            cv2.imwrite(f"tmp/{dcm_path[dcm_path.rfind(os.path.sep) + 1:dcm_path.rfind('.')]}_segmented.png", curr_masks.squeeze(0))
            mask = (mask * pred_idx + curr_masks) / (pred_idx + 1)

    # return mask.squeeze(0)

if __name__ == '__main__':
    args = argparser()
    mask = predict(args)
    # dest_path = args['dcm_path'][:args['dcm_path'].rfind('.')] + '_segmented.png' 
    # cv2.imwrite(dest_path, mask)
    # print(f'Result is stored in {dest_path}')
import argparse
import os

import albumentations as albu
import torch

import cv2
import numpy as np
import pydicom
from pathlib import Path
import importlib

from albumentations.pytorch.transforms import ToTensor
from skimage.transform import resize

from utils.mask_functions import mask2rle
from utils.helpers import load_yaml

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dcm-path', type=str, help='Path to image')
    parser.add_argument('--config', type=str, help='Path to config file path')
    return vars(parser.parse_args())

def build_checkpoints_list(cfg):
    usefolds = cfg['USEFOLDS']
    checkpoints_list = []
    for fold_id in usefolds:
        filename = f"{cfg['CHECKPOINTS']['PIPELINE_NAME']}_fold_{fold_id}.pth"
        checkpoints_list.append(Path(cfg['CHECKPOINTS']['BEST_FOLDER'], filename))
    
    return checkpoints_list

def apply_thresholds(mask, n_objects, area_threshold, top_score_threshold, 
                     bottom_score_threshold, leak_score_threshold):
    if n_objects == 1:
        # crazy_mask = (mask > top_score_threshold).astype(np.uint8)
        # if crazy_mask.sum() < area_threshold: 
        #     return -1
        mask = (mask > bottom_score_threshold).astype(np.uint8)
    else:
        mask = (mask > leak_score_threshold).astype(np.uint8)

    choosen = mask * 255

    if mask.shape[0] == 1024:
        reshaped_mask = choosen
    else:
        reshaped_mask = cv2.resize(
            choosen,
            dsize=(1024, 1024),
            interpolation=cv2.INTER_LINEAR
        )
    reshaped_mask = ((reshaped_mask > 127) * 255).astype('uint8')
    return reshaped_mask

def predict(dcm_path, cfg):
    image = pydicom.read_file(dcm_path).pixel_array
    image = resize(image, (cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE']))
    image = (image * 255).astype('uint8')
    image = np.dstack([image] * 3)
    
    fn = dcm_path[:dcm_path.rfind('.')]
    cv2.imwrite(fn + '.png', image)
    print(f'DCM file is trasformed to PNG in {fn}.png')

    # model = AlbuNet(pretrained=False).to(cfg['DEVICE'])
    module = importlib.import_module(cfg['MODEL']['PY'])
    model_class = getattr(module, cfg['MODEL']['CLASS'])
    model = model_class(**cfg['MODEL'].get('ARGS', None)).to(cfg['DEVICE'])
    
    transform = albu.load(cfg['TRANSFORMS'])

    to_tensor = ToTensor()
    sample = transform(image=image)
    sample = to_tensor(**sample)
    image = sample['image'].unsqueeze(0).to(cfg['DEVICE'])

    checkpoints_list = build_checkpoints_list(cfg)
    mask = 0
    for pred_idx, checkpoint_path in enumerate(checkpoints_list):
        print(checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(cfg['DEVICE'])))
        model.eval()

        preds = model(image)
        curr_masks = torch.sigmoid(preds)
        curr_masks = curr_masks.squeeze(1).cpu().detach().numpy()
        mask = (mask * pred_idx + curr_masks) / (pred_idx + 1)
    # return (mask.squeeze(0) * 255).astype('uint8')

    area_threshold = cfg['AREA_THRESHOLD']
    top_score_threshold = cfg['TOP_SCORE_THRESHOLD']
    bottom_score_threshold = cfg['BOTTOM_SCORE_THRESHOLD']
    if cfg['USELEAK']:
        leak_score_threshold = cfg['LEAK_SCORE_THRESHOLD']
    else:
        leak_score_threshold = bottom_score_threshold

    return apply_thresholds(
        mask.squeeze(0), 1,
        area_threshold, top_score_threshold, bottom_score_threshold,
        leak_score_threshold
    )

if __name__ == '__main__':
    args = argparser()
    assert Path(args['dcm_path']).is_file() and args['dcm_path'][-3:] == 'dcm', 'image path is invalid'

    config_path = Path(args['config'].strip('/'))
    inference_config = load_yaml(config_path)

    mask = predict(args['dcm_path'], inference_config)
    dest_path = args['dcm_path'][:args['dcm_path'].rfind('.')] + '_segmented.png' 
    cv2.imwrite(dest_path, mask)
    print(f'Result is stored in {dest_path}')

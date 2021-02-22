import argparse
import os

import albumentations as albu
import torch

import cv2
import numpy as np
import pydicom
from pathlib import Path

from models.ternausnet import AlbuNet
from albumentations.pytorch.transforms import ToTensor
from skimage.transform import resize

from utils.mask_functions import mask2rle
from utils.helpers import load_yaml

def build_checkpoints_list(cfg):
    usefolds = cfg['USEFOLDS']
    checkpoints_list = []
    for fold_id in usefolds:
        filename = f"{cfg['CHECKPOINTS']['PIPELINE_NAME']}_fold_{fold_id}.pth"
        checkpoints_list.append(Path(cfg['CHECKPOINTS']['BEST_FOLDER'], filename))
    
    return checkpoints_list

cfg_path = r"experiments\albunet_valid\prediction.yaml"
cfg = load_yaml(cfg_path)
model = AlbuNet(pretrained=False).to(cfg['DEVICE'])
transform = albu.load(cfg['TRANSFORMS'])
checkpoint_path = build_checkpoints_list(cfg)[0]
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(cfg['DEVICE'])))
model.eval()
print("Load model")




def apply_thresholds(mask, n_objects, area_threshold, top_score_threshold, 
                     bottom_score_threshold, leak_score_threshold):
    if n_objects == 1:
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

def predict(dcm_path):
    image = pydicom.read_file(dcm_path).pixel_array
    image = resize(image, (cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE']))
    image = (image * 255).astype('uint8')
    image = np.dstack([image] * 3)
    
    fn = dcm_path[:dcm_path.rfind('.')]
    cv2.imwrite(fn + '.png', image)
    print(f'DCM file is trasformed to PNG in {fn}.png')

    to_tensor = ToTensor()
    sample = transform(image=image)
    sample = to_tensor(**sample)
    imageT = sample['image'].unsqueeze(0).to(cfg['DEVICE'])

    mask = 0
    pred_idx = 0


    preds = model(imageT)
    curr_masks = torch.sigmoid(preds)
    curr_masks = curr_masks.squeeze(1).cpu().detach().numpy()
    mask = (mask * 0 + curr_masks) / (pred_idx + 1)

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
    ),image,fn + '.png'
def test(dcm_path):
    mask,image,path_ori = predict(dcm_path)
    dest_path = dcm_path[:dcm_path.rfind('.')] + '_segmented.png' 
    cv2.imwrite(dest_path, mask)
    dest_path1 = dcm_path[:dcm_path.rfind('.')] + '_segmentedT.png' 
    image[:,:,0][mask==255]=255
    image[:,:,1][mask==255]=255
    image[:,:,2][mask==255]=255
    cv2.imwrite(dest_path1,image)
    print(f'Result is stored in {dest_path}')
    return ".png",'_segmented.png','_segmentedT.png'

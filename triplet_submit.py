import argparse
import pickle
from tqdm import tqdm
from pathlib import Path

import cv2

import numpy as np
import pandas as pd
from collections import defaultdict

from utils.mask_functions import mask2rle
from utils.helpers import load_yaml

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file path')
    return vars(parser.parse_args())

def extract_largest(mask, n_objects):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    contours = np.array(contours)[np.argsort(areas)[::-1]]
    background = np.zeros(mask.shape, 'uint8')
    chosen = cv2.drawContours(background, contours[:n_objects], -1, 255, -1)
    return chosen

def remove_smallest(mask, min_contour_area):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    background = np.zeros(mask.shape, 'uint8')
    chosen = cv2.drawContours(background, contours, -1, 255, -1)
    return chosen

def apply_threshold(mask, n_objects, area_threshold, top_score_threshold,
        bottom_score_threshold, leak_score_threshold, use_contours, min_contour_area):
    if n_objects == 1:
        crazy_mask = mask > top_score_threshold
        if crazy_mask.sum() < area_threshold:
            return -1
        mask = (mask > bottom_score_threshold).astype('uint8')
    else:
        mask = (mask > leak_score_threshold).astype('uint8')

    if min_contour_area > 0:
        chosen = remove_smallest(mask, min_contour_area)
    elif use_contours:
        chosen = extract_largest(mask, n_objects)
    else:
        chosen = mask * 255

    if mask.shape[0] == 1024:
        reshaped_mask = chosen
    else:
        reshaped_mask = cv2.resize(chosen, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    reshaped_mask = (reshaped_mask > 127).astype('int') * 255
    return mask2rle(reshaped_mask.T, 1024, 1024)

def build_rle_dict(mask_dict, n_objects_dict, area_threshold, top_score_threshold,
            bottom_score_threshold, leak_score_threshold, use_contours, min_contour_area):
    
    rle_dict = {}
    for name, mask in tqdm(mask_dict.items()):
        if name not in n_objects_dict:
            continue
        n_objects = n_objects_dict[name]
        rle_dict[name] = apply_threshold(mask, n_objects, area_threshold,
                top_score_threshold, bottom_score_threshold, leak_score_threshold,
                use_contours, min_contour_area)
    return rle_dict

def build_submission(rle_dict, sample_sub):
    sub = pd.DataFrame.from_dict([rle_dict]).T.reset_index()
    sub.columns = sample_sub.columns
    sub.loc[sub.EncodedPixels == '', 'EncodedPixels'] = -1
    return sub

def load_mask_dict(config):
    reshape_mode = config.get('')
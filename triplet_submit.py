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

def extract_larger(mask, n_objects):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    contours = np.array(contours)[np.argsort(areas)[::-1]]
    background = np.zeros(mask.shape, 'uint8')
    chosen = cv2.drawContours(background, contours[:n_objects], -1, 255, -1)
    return chosen

def remove_smallest(mask, min_contour_area):
    
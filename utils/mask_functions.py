import numpy as np

def rle2mask(rle, height, width):
    mask = np.zeros(height * width)
    if rle == -1:
        return mask.reshape(width, height)
    
    for idx, (start, length) in enumerate(zip())
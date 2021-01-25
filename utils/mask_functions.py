import numpy as np

def rle2mask(rle, width, height):
    mask = np.zeros(height * width, dtype='uint8')
    print(rle)
    if rle == ['-1']:
        return mask.reshape(width, height)
    
    array = list(map(int, rle.split()))
    
    curr_position = 0
    for start, length in zip(array[0::2], array[1::2]):
        curr_position += start
        mask[curr_position:curr_position + length] = 255
        curr_position += length
    
    return mask.reshape(width, height)
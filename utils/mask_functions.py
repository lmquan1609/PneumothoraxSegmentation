import numpy as np

def rle2mask(rle, width, height):
    mask = np.zeros(height * width, dtype='uint8')
    if rle == ['-1']:
        return mask.reshape(width, height)
    
    array = list(map(int, rle.split()))
    
    curr_position = 0
    for start, length in zip(array[0::2], array[1::2]):
        curr_position += start
        mask[curr_position:curr_position + length] = 255
        curr_position += length
    
    return mask.reshape(width, height)

def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1

    return " ".join(rle)
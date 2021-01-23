from imutils import paths
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import argparse
import os
import shutil
import pydicom

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', default=os.path.join('siim', 'dicom-images-train'), type=str, help='Directory to training path')
    parser.add_argument('--test-path', default=os.path.join('siim', 'dicom-images-test'), type=str, help='Directory to testing path')
    parser.add_argument('--rle-path', default=os.path.join('siim', 'train-rle.csv'), type=str, help='Path to rle csv file')
    parser.add_argument('--output-path', default='dataset', type=str, help='Path for saving dataset')
    parser.add_argument('--image-size', default=1024, type=int, help='Converted image size')
    parser.add_argument('--number-threads', default=4, type=int, help='Number of using threads')
    return parser.parse_args()

def get_mask(encode, height, width):
    if encode == []:
        encode.append(-1)
    mask = rle2mask(encode[0], height, width)
    for e in encode[1:]:
        mask += rle2mask(e, height, width)
    
    return mask.T

def conversion_for_train_file(file_name, encode_df, output_path, image_size):
    image = pydicom.read_file(file_name).pixel_array
    fn_wo_ext = file_name.split(os.path.sep)[-1][:-4]
    encode = list(encode_df.loc[encode_df['ImageId'] == fn_wo_ext, ' EncodedPixels'])
    encode = get_mask(encode, image.shape[0], image.shape[1])

def conversion_for_train(train_fns, encode_df, output_path, image_size, n_threads=-1):
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'mask'), exist_ok=True)

    try:
        Parallel(n_jobs=n_threads, backend='threading')(delayed(conversion_for_train_file)(
            fn, encode_df, output_path, image_size) for fn in tqdm(train_fns))
    except pydicom.errors.InvalidDicomError:
        print('InvalidDicomError')

if __name__ == '__main__':
    args = argparser()
    train_fns = sorted(paths.list_files(args['train_path']))
    test_fns = sorted(paths.list_files(args['test_path']))
    rle = pd.read_csv(args['rle_path'])

    conversion_for_train(train_fns, rle, args['output_path'], args['image_size'], n_threads=args['number_threads'])
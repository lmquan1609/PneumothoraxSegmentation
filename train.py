import argparse
import os

from pathlib import Path
from utils import helpers
import albumentations as albu

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-config', type=str, help='Path to train config file path')
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = argparser()
    config_file = Path(args['train_config'].strip('/'))
    experiment_folder = config_file.parents[0]

    train_config = helpers.load_yaml(config_file)

    log_dir = Path(experiment_folder, train_config['LOGGER_DIR'])
    log_dir.mkdir(parents=True, exist_ok=True)

    main_logger = helpers.init_logger(log_dir, 'train_main.log')

    seed = train_config['SEED']
    helpers.init_seed(seed)
    main_logger.info(train_config)

    if "DEVICE_LIST" in train_config:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, train_config['DEVICE_LIST']))

    pipeline_name = train_config['PIPELINE_NAME']
    dataset_foder = train_config['DATA_DIRECTORY']

    train_transform = albu.load(train_config['TRAIN_TRANSFORMS'])
    valid_transform = albu.load(train_config['VALID_TRANSFORMS'])

    # ? non_empty_mask_proba/ use_sampler is not sure
    non_empty_mask_proba = train_config.get('NON_EMPTY_MASK_PROBA', 0)
    use_sampler = train_config['USE_SAMPLER']

    dataset_folder = train_config['DATA_DIRECTORY']
    folds_path = train_config['FOLD']['FILE']

    num_workers = train_config['WORKERS']
    batch_size = train_config['BATCH_SIZE']
    n_folds = train_config['FOLD']['NUMBER']

    
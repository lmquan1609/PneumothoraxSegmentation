import argparse
import os
import torch

from pathlib import Path
from utils import helpers
import albumentations as albu

import importlib
import functools

from pneumothorax_dataset import PneumothoraxDataset, PneumoSampler

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-config', type=str, help='Path to train config file path')
    return vars(parser.parse_args())

def train_fold(train_config, experiment_folder, pipeline_name, log_dir, fold_id, 
            train_dataloader, val_dataloader, binarizer_fn, eval_fn):

        fold_logger = helpers.init_logger(log_dir, f'train_log_{fold_id}.log')

        best_checkpoint_folder = Path(experiment_folder, train_config['CHECKPOINTS']['BEST_FOLDER'])
        best_checkpoint_folder.mkdir(parents=True, exist_ok=True)

        checkpoint_history_folder = Path(
            experiment_folder,
            train_config['CHECKPOINTS']['FULL_FOLDER'],
            f'fold{fold_id}'
        )
        checkpoint_history_folder.mkdir(parents=True, exist_ok=True)
        checkpoints_topk = train_config['CHECKPOINTS']['TOPK']

        calculation_name = f'{pipeline_name}_fold{fold_id}'

        device = train_config['DEVICE']

        

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
    dataset_folder = train_config['DATA_DIRECTORY']

    train_transform = albu.load(train_config['TRAIN_TRANSFORMS'])
    val_transform = albu.load(train_config['VALID_TRANSFORMS'])

    # ? non_empty_mask_prob/ use_sampler is not sure
    non_empty_mask_prob = train_config.get('NON_EMPTY_MASK_PROB', 0)
    use_sampler = train_config['USE_SAMPLER']

    dataset_folder = train_config['DATA_DIRECTORY']
    folds_distr_path = train_config['FOLD']['FILE']

    num_workers = train_config['WORKERS']
    batch_size = train_config['BATCH_SIZE']
    n_folds = train_config['FOLD']['NUMBER']

    usefolds = map(str, train_config['FOLD']['USEFOLDS'])

    binarizer_module = importlib.import_module(train_config['MASK_BINARIZER']['PY'])
    binarizer_class = getattr(binarizer_module, train_config['MASK_BINARIZER']['CLASS'])
    binarizer_fn = binarizer_class(**train_config['MASK_BINARIZER']['ARGS'])

    eval_module = importlib.import_module(train_config['EVALUATION_METRIC']['PY'])
    eval_fn = getattr(eval_module, train_config['EVALUATION_METRIC']['CLASS'])
    eval_fn = functools.partial(**train_config['EVALUATION_METRIC']['ARGS'])

    for fold_id in usefolds:
        main_logger.info(f'Start training of {fold_id} fold...')

        train_dataset = PneumothoraxDataset(
            data_folder=dataset_folder, mode='train',
            train_transform=train_transform, folder_index=fold_id,
            folds_distr_path=folds_distr_path
        )
        train_sampler = PneumoSampler(folds_distr_path, fold_id, non_empty_mask_prob)
        if use_sampler:
            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=batch_size,
                sampler=train_sampler, num_workers=num_workers
            )
        else:
            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=batch_size,
                sampler=train_sampler, shuffle=True
            )

        val_dataset = PneumothoraxDataset(
            data_folder=dataset_folder, mode='val',
            transform=val_transform, folder_index=fold_id,
            folds_distr_path=folds_distr_path
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, num_workers=num_workers
        )


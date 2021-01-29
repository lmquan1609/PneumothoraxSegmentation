import argparse
import os
import torch

from pathlib import Path
from utils import helpers
import albumentations as albu

import importlib
import functools

from pneumothorax_dataset import PneumothoraxDataset, PneumoSampler
from learning import Learning

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-config', type=str, help='Path to train config file path')
    return vars(parser.parse_args())

def train_fold(train_config, experiment_folder, pipeline_name, log_dir, fold_id, 
            train_dataloader, val_dataloader, binarizer_fn, eval_fn):

    fold_logger = helpers.init_logger(log_dir, f'train_log_{fold_id}.log')

    best_checkpoint_folder = Path(experiment_folder, train_config['CHECKPOINTS']['BEST_FOLDER'])
    best_checkpoint_folder.mkdir(parents=True, exist_ok=True)

    checkpoints_history_folder = Path(
        experiment_folder,
        train_config['CHECKPOINTS']['FULL_FOLDER'],
        f'fold_{fold_id}'
    )
    checkpoints_history_folder.mkdir(parents=True, exist_ok=True)
    checkpoints_topk = train_config['CHECKPOINTS']['TOPK']

    calculation_name = f'{pipeline_name}_fold_{fold_id}'

    device = train_config['DEVICE']

    module = importlib.import_module(train_config['MODEL']['PY'])
    model_class = getattr(module, train_config['MODEL']['CLASS'])
    model = model_class(**train_config['MODEL']['ARGS'])

    pretrained_model_config = train_config['MODEL'].get('PRETRAINED', False)
    if pretrained_model_config:
        loaded_pipeline_name = pretrained_model_config['PIPELINE_NAME']
        pretrained_model_path = Path(
            pretrained_model_config['PIPELINE_PATH'],
            pretrained_model_config['CHECKPOINTS_FOLDER'],
            f'{loaded_pipeline_name}_fold_{fold_id}.pth'
        )

        if pretrained_model_path.is_file():
            model.load_state_dict(torch.load(pretrained_model_path))
            fold_logger.info(f'Load model from {pretrained_model_path}')

    if len(train_config['DEVICE_LIST']) > 1:
        model = torch.nn.DataParallel(model)
        
    module = importlib.import_module(train_config['CRITERION']['PY'])
    loss_class = getattr(module, train_config['CRITERION']['CLASS'])
    loss_fn = loss_class(**train_config['CRITERION']['ARGS'])

    optimizer_class = getattr(torch.optim, train_config['OPTIMIZER']['CLASS'])
    optimizer = optimizer_class(model.parameters(), **train_config['CRITERION']['ARGS'])
    scheduler_class = getattr(torch.optim.lr_scheduler, train_config['SCHEDULER']['CLASS'])
    scheduler = scheduler_class(optimizer, **train_config['SCHEDULER']['ARGS'])

    n_epochs = train_config['EPOCHS']
    grad_clip = train_config['GRADIENT_CLIPPING']
    grad_accum = train_config['GRADIENT_ACCUMULATION_STEPS']
    early_stopping = train_config['EARLY_STOPPING']
    validation_frequency = train_config.get('VALIDATION_FREQUENCY', 1)

    freeze_model = train_config['MODEL']['FREEZE']

    Learning(
        optimizer, binarizer_fn, loss_fn, eval_fn, device, n_epochs, scheduler,
        freeze_model, grad_clip, grad_accum, early_stopping, validation_frequency,
        calculation_name, best_checkpoint_folder, checkpoints_history_folder,
        checkpoints_topk, fold_logger
    ).run_train(model, train_dataloader, val_dataloader)

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
    val_transform = albu.load(train_config['VAL_TRANSFORMS'])

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

        train_fold(
            train_config, experiment_folder, pipeline_name, log_dir, fold_id,
            train_dataloader, val_dataloader, binarizer_fn, eval_fn
        )
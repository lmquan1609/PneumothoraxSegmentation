import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path

import heapq
from collections import defaultdict

class Learning:
    def __init__(self, optimizer, binarizer_fn, loss_fn, eval_fn, device, n_epochs, scheduler,
                freeze_model, grad_clip, grad_accum, early_stopping, validation_frequency,
                calculation_name, best_checkpoint_folder, checkpoints_history_folder,
                checkpoints_topk, logger):
        self.logger = logger

        self.optimizer = optimizer
        self.binarizer_fn = binarizer_fn
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn

        self.device = device
        self.n_epochs = n_epochs
        self.scheduler = scheduler
        self.freeze_model = freeze_model
        self.grad_clip = grad_clip
        self.grad_accum = grad_accum
        self.early_stopping = early_stopping
        self.validation_frequency = validation_frequency

        self.calculation_name = calculation_name
        self.best_checkpoint_path = Path(
            best_checkpoint_folder,
            f'{self.calculation_name}.pth'
        )
        self.checkpoints_history_folder = Path(checkpoints_history_folder)
        self.checkpoints_topk = checkpoints_topk
        self.score_heap = []
        self.summary_file = Path(self.checkpoints_history_folder, 'summary.csv')
        if self.summary_file.is_file():
            self.best_score = pd.read_csv(self.summary_file)['best_metric'].max()
            logger.info(f'Pretrained best score is {self.best_score:.5}')
        else:
            self.best_score = 0
        
        self.best_epoch = -1

    def train_epoch(self, model, loader):
        curr_loss_mean = 0
        tqdm_loader = tqdm(loader)

        for batch_idx, (images, labels) in enumerate(tqdm_loader):
            loss, preds = self.batch_train(model, images, labels, batch_idx)

            # cummulative moving average
            curr_loss_mean = (curr_loss_mean * batch_idx + loss) / (batch_idx + 1)

            tqdm_loader.set_description(f"loss: {curr_loss_mean:.4} at lr: {self.optimizer.param_groups[0]['lr']}")
        
        return curr_loss_mean

    def batch_train(self, model, batch_images, batch_labels, batch_idx):
        batch_images, batch_labels = batch_images.to(self.device), batch_labels.to(self.device)
        preds = model(batch_images)
        loss = self.loss_fn(preds, batch_labels)

        loss.backward()
        if batch_idx % self.grad_accum == self.grad_accum - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item(), preds
    
    def val_epoch(self, model, loader):
        tqdm_loader = tqdm(loader)
        curr_score_mean = 0
        used_thresholds = self.binarizer_fn.thresholds
        metrics = defaultdict(float)

        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm_loader):
                pred_probs = self.batch_val(model, images)
                labels = labels.to(device)
                mask_generator = self.binarizer_fn.transform(pred_probs)

                for curr_threshold, curr_mask in zip(used_thresholds, mask_generator):
                    curr_metric = self.eval_fn(curr_mask, labels).item()
                    curr_threshold = tuple(curr_threshold)
                    metrics[curr_threshold] = (metrics[curr_threshold] * batch_idx + curr_metric) / (batch_idx + 1)
                
                best_threshold = max(metrics, key=metrics.get)
                tqdm_loader.set_description(f'score: {metrics[best_threshold]:.5} at threshold {best_threshold}')
        
        return metrics, metrics[best_threshold]

    def batch_val(self, model, batch_image):
        batch_image = batch_image.to(self.device)
        preds = model(batch_image)
        return torch.sigmoid(preds)
    
    def process_summary(self, metrics, epoch):
        best_threshold = max(metrics, key=metrics.get)

        epoch_summary = pd.DataFrame.from_dict([metrics])
        epoch_summary['epoch'] = epoch
        epoch_summary['best_metric'] = metrics[best_threshold]
        epoch_summary
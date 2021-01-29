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
            best_checkpoint_folder, f'{self.calculation_name}.pth'
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
                tqdm_loader.set_description(f'Score: {metrics[best_threshold]:.5} at threshold {best_threshold}')
        
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
        epoch_summary = epoch_summary[['epoch', 'best_metric'] + list(metrics.keys())]
        epoch_summary.columns = list(map(str, epoch_summary.columns))

        self.logger.info(f'Epoch {epoch + 1}\tScore: {metrics[best_threshold]:.5} at params: {best_threshold}')

        if not self.summary_file.is_file():
            epoch_summary.to_csv(self.summary_file, index=False)
        else:
            summary = pd.read_csv(self.summary_file)
            summary = summary.append(epoch_summary).reset_index(drop=True)
            summary.to_csv(self.summary_file, index=False)

    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict

    def post_preprocessing(self, score, epoch, model):
        if self.freeze_model:
            return
        
        checkpoints_history_path = Path(
            self.checkpoints_history_folder, f'{self.calculation_name}_epoch_{epoch:03d}.pth'
        )

        torch.save(self.get_state_dict(model), checkpoints_history_path)
        heapq.heappush(self.score_heap, (score, checkpoints_history_path))
        if len(self.score_heap) > self.checkpoints_topk:
            _, removing_checkpoint_path = heapq.heappop(self.score_heap)
            removing_checkpoint_path.unlink()
            self.logger.info(f'Removed checkpint at {removing_checkpoint_path}')
        
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            torch.save(self.get_state_dict(model), self.best_checkpoint_path)
            self.logger.info(f'Epoch {epoch + 1}:\tBest model\tScore: {score:.5}')

        if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.scheduler.step(score)
        else:
            self.scheduler.step()

    def run_train(self, model, train_dataloader, val_dataloader):
        model.to(self.device)
        for epoch in range(self.n_epochs):
            if not self.freeze_model:
                self.logger.info(f'Epoch {epoch + 1}\tStart training...')
                model.train()
                train_loss_mean = self.train_epoch(model, train_dataloader)
                self.logger.info(f'Epoch {epoch + 1}\tCalculated train loss: {train_loss_mean:.5}')

            if epoch % self.validation_frequency != (self.validation_frequency - 1):
                self.logger.info('Skip validation...')
                continue
                
            self.logger.info(f'Epoch {epoch + 1}\tStart validation...')
            model.eval()
            metrics, score = self.val_epoch(model, val_dataloader)

            self.process_summary(metrics, epoch)

            self.post_preprocessing(score, epoch, model)

            if epoch - self.best_epoch > self.early_stopping:
                self.logger.info('EARLY STOPPING')
                break
        
        return self.best_score, self.best_epoch
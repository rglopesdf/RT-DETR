import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from typing import Dict

from src.misc import dist
from src.core import BaseConfig


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.last_epoch = cfg.last_epoch
        self.best_metric = None  # Initialize best metric
        self.model = dist.warp_model(cfg.model.to(self.device), cfg.find_unused_parameters, cfg.sync_bn)
        self.criterion = cfg.criterion.to(self.device)
        self.postprocessor = cfg.postprocessor

        # NOTE: Should load tuning state before EMA instance building
        if self.cfg.tuning:
            print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        self.scaler = cfg.scaler
        self.ema = cfg.ema.to(self.device) if cfg.ema is not None else None

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self):
        '''Avoid instantiating unnecessary classes'''
        pass  # Setup is handled in __init__

    def train(self):
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler

        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.resume(self.cfg.resume)

        self.train_dataloader = dist.warp_loader(self.cfg.train_dataloader, shuffle=self.cfg.train_dataloader.shuffle)
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle)
        self.fit()  # Start the training loop

    def eval(self):
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle)

        if self.cfg.resume:
            print(f'Resume from {self.cfg.resume}')
            self.resume(self.cfg.resume)

        self.validate(self.last_epoch)

    def state_dict(self):
        '''Return state dictionary for checkpointing'''
        state = {
            'model': dist.de_parallel(self.model).state_dict(),
            'date': datetime.now().isoformat(),
            'last_epoch': self.last_epoch
        }

        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()

        if self.ema is not None:
            state['ema'] = self.ema.state_dict()

        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()

        return state

    def load_state_dict(self, state):
        '''Load state dictionary from checkpoint'''
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Loading last_epoch')

        if 'model' in state:
            if dist.is_parallel(self.model):
                self.model.module.load_state_dict(state['model'])
            else:
                self.model.load_state_dict(state['model'])
            print('Loading model.state_dict')

        if 'ema' in state and self.ema is not None:
            self.ema.load_state_dict(state['ema'])
            print('Loading ema.state_dict')

        if 'optimizer' in state and self.optimizer is not None:
            self.optimizer.load_state_dict(state['optimizer'])
            print('Loading optimizer.state_dict')

        if 'lr_scheduler' in state and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            print('Loading lr_scheduler.state_dict')

        if 'scaler' in state and self.scaler is not None:
            self.scaler.load_state_dict(state['scaler'])
            print('Loading scaler.state_dict')

    def save_checkpoint(self, name):
        '''Save checkpoint with given name'''
        path = self.output_dir / f'checkpoint_{name}.pth'
        state = self.state_dict()
        dist.save_on_master(state, path)
        print(f'Saved checkpoint: {path}')

    def resume(self, path):
        '''Resume training from checkpoint'''
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state)

    def load_tuning_state(self, path):
        '''Load model state for fine-tuning'''
        if 'http' in path:
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        module = dist.de_parallel(self.model)

        # Load EMA weights if available
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        print(f'Loaded model.state_dict with info: {infos}')

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}

    def fit(self):
        '''Training loop with checkpoint saving'''
        num_epochs = self.cfg.num_epochs
        for epoch in range(self.last_epoch + 1, num_epochs):
            self.last_epoch = epoch
            self.train_one_epoch(epoch)

            # Perform validation
            metric = self.validate(epoch)

            # Check if current metric is better than the best so far
            if self.best_metric is None or metric > self.best_metric:
                self.best_metric = metric
                self.save_checkpoint('best')  # Save best checkpoint

            self.save_checkpoint('last')  # Save last checkpoint

            # Optionally, update learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def train_one_epoch(self, epoch):
        '''Train the model for one epoch'''
        self.model.train()
        for batch in self.train_dataloader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Update EMA if available
            if self.ema is not None:
                self.ema.update(self.model)

    def validate(self, epoch):
        '''Validate the model and return the metric of interest (AP50)'''
        self.model.eval()
        total_metric = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                predictions = self.postprocessor(outputs)

                # Compute AP50 metric
                metric = self.compute_ap50(predictions, targets)
                total_metric += metric
                num_batches += 1

        average_metric = total_metric / num_batches
        print(f'Epoch {epoch}: AP50 = {average_metric:.4f}')
        return average_metric

    def compute_ap50(self, predictions, targets):
        '''Compute the AP50 metric'''
        # Implement AP50 calculation based on your evaluation code
        # This is a placeholder implementation
        ap50 = 0.0  # Replace with actual AP50 computation
        return ap50

    def val(self):
        '''Validation loop (if needed separately)'''
        self.validate(self.last_epoch)
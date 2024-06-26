import os
import time
from typing import Optional, Tuple, Mapping, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import Optimizer

import models.optimizer as optim
from datasets.loader import get_train_dataloader, get_val_dataloader, get_test_dataloader
from utils.misc import mkdir
from utils.meters import AverageMeter, ProgressMeter


class Trainer:
    def __init__(
            self,
            cfg,
            model,
            metric_names: Tuple[str],
            loss_names: Tuple[str],
            optimizer: Optional[Union[Optimizer, List[Optimizer]]] = None
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer

        assert len(metric_names) > 0 and len(loss_names) > 0
        self.metric_names = metric_names
        self.loss_names = loss_names

        self.cur_epoch = 0
        self.cur_iter = 0

        # Create the train and val (test) loaders.
        self.train_loader = get_train_dataloader(self.cfg)
        self.val_loader = get_val_dataloader(self.cfg)
        self.test_loader = get_test_dataloader(self.cfg)

        # create optimizer
        if self.optimizer is None:
            self.create_optimizer()

    def create_optimizer(self):
        self.optimizer = optim.construct_optimizer(self.model, self.cfg)

    def train(self):
        best_metric = self.cfg.TRAIN.BEST_METRIC_INITIAL
        for cur_epoch in range(self.cfg.SOLVER.START_EPOCH, self.cfg.SOLVER.MAX_EPOCH):
            self.train_epoch()

            # Evaluate the model on validation set.
            if self._is_eval_epoch(cur_epoch):
                tracking_meter = self.eval_epoch()
                # check improvement
                is_best = self._check_improvement(tracking_meter.avg, best_metric)
                # Save a checkpoint on improvement.
                if is_best:
                    with open(mkdir(self.cfg.RESULT_DIR) / "best_result.txt", 'w') as f:
                        f.write(f"Val/{tracking_meter.name}: {tracking_meter.avg}\tEpoch: {self.cur_epoch}")
                    self.save_best_model()
                    best_metric = tracking_meter.avg
            self.cur_epoch += 1

    def _check_improvement(self, cur_metric, best_metric):
        if (self.cfg.TRAIN.BEST_LOWER and cur_metric < best_metric) \
                or (not self.cfg.TRAIN.BEST_LOWER and cur_metric > best_metric):
            return True
        else:
            return False

    def train_epoch(self):
        #! 여기 설명 필요함
        # set meters
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        metric_meters = self._get_metric_meters()
        loss_meters = self._get_loss_meters()
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, *metric_meters, *loss_meters],
            prefix="Epoch: [{}]".format(self.cur_epoch)
        )

        # switch to train mode
        self.model.train()

        data_size = len(self.train_loader)

        start = time.time()

        for cur_iter, inputs in enumerate(self.train_loader):
            self.cur_iter = cur_iter
            # dictionary for logging values
            log_dict = {}

            # measure data loading time
            data_time.update(time.time() - start)

            # Update the learning rate.
            lr = optim.get_epoch_lr(self.cur_epoch + float(cur_iter) / data_size, self.cfg)
            optim.set_lr(self.optimizer, lr)

            # log to W&B
            log_dict.update({
                "lr/": lr
            })

            outputs = self.train_step(inputs)

            # update metric and loss meters, and log to W&B
            batch_size = self._find_batch_size(inputs)
            self._update_metric_meters(metric_meters, outputs["metrics"], batch_size)
            self._update_loss_meters(loss_meters, outputs["losses"], batch_size)
            log_dict.update({
                f"Train/{metric_meter.name}": metric_meter.val for metric_meter in metric_meters
            })
            log_dict.update({
                f"Train/Loss_Total": loss_meter.val for loss_meter in loss_meters
            })
            ############################    MUC    ############################
            for i in range(len(outputs['weights'])):    
                for j in range(len(outputs['weights'][i])):
                    log_dict.update({
                        f"Train/Layer_{i}_Weight_{j}": outputs["weights"][i][j]
                    })
            log_dict.update({
                "Train/Loss_MSE": outputs["loss_mse"]
            })
            log_dict.update({
                "Train/Loss_MACs": outputs["loss_MACs"]
            })
            log_dict.update({
                "Train/Loss_LASSO": outputs["loss_LASSO"]
            })
            ############################    MUC    ############################

            if cur_iter % self.cfg.TRAIN.PRINT_FREQ == 0:
                progress.display(cur_iter)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if self.cfg.WANDB.ENABLE:
                wandb.log(log_dict)

        log_dict = {}

        # log to W&B
        if self.cfg.WANDB.ENABLE:
            wandb.log(log_dict, commit=False)

    def _get_metric_meters(self):
        return [AverageMeter(metric_name, ":.3f") for metric_name in self.metric_names]

    def _get_loss_meters(self):
        return [AverageMeter(f"Loss {loss_name}", ":.4e") for loss_name in self.loss_names]

    @staticmethod
    def _update_metric_meters(metric_meters, metrics, batch_size):
        assert len(metric_meters) == len(metrics)
        for metric_meter, metric in zip(metric_meters, metrics):
            metric_meter.update(metric.item(), batch_size)

    @staticmethod
    def _update_loss_meters(loss_meters, losses, batch_size):
        assert len(loss_meters) == len(losses)
        for loss_meter, loss in zip(loss_meters, losses):
            loss_meter.update(loss.item(), batch_size)
    
    #* training 변경하려면 수정해야 되는 부분
    def train_step(self, inputs):
        # override for different methods
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        
        ground_truth = dec_window[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:].float()
        dec_zeros = torch.zeros_like(dec_window[:, -self.cfg.DATA.PRED_LEN:, :]).float()
        dec_window = torch.cat([dec_window[:, :self.cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float().cuda()
        
        # model_cfg = getattr(self.cfg.MODEL, self.cfg.MODEL_NAME.upper())
        model_cfg = self.cfg.MODEL
       
        if model_cfg.output_attention:
            pred, weights = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)[0]
        else:
            pred, weights = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)
        
        pred = pred[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:]
                   
        ############################    MUC    ############################
        loss_mse = F.mse_loss(pred, ground_truth)
        loss_MACs = self.cfg.TRAIN.MACs_weight * self.model.MACs()
        loss_LASSO = self.cfg.TRAIN.LASSO_weight * self.model.LASSO()
        loss = loss_mse + loss_MACs + loss_LASSO        
        ############################    MUC    ############################
        metric = F.l1_loss(pred, ground_truth)
        
        if type(self.optimizer).__name__ == 'SAM':
            loss.backward()
            self.optimizer.first_step(zero_grad=True)
            
            if model_cfg.output_attention:
                pred, weights = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)[0]
            else:
                pred, weights = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)
            
            ############################    MUC    ############################
            loss_mse = F.mse_loss(pred, ground_truth)
            loss_MACs = self.cfg.TRAIN.MACs_weight * self.model.MACs()
            loss_LASSO = self.cfg.TRAIN.LASSO_weight * self.model.LASSO()
            loss = loss_mse + loss_MACs + loss_LASSO
            ############################    MUC    ############################
        
            metric = F.l1_loss(pred, ground_truth)
            
            loss.backward()
            self.optimizer.second_step(zero_grad=True)
            
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        outputs = dict(
            losses=(loss,),
            metrics=(metric,),
            ############################    MUC    ############################
            weights = weights,
            loss_mse = loss_mse,
            loss_MACs = loss_MACs,
            loss_LASSO = loss_LASSO
            ############################    MUC    ############################
        )

        return outputs

    def _load_from_checkpoint(self):
        pass

    def _find_batch_size(self, inputs):
        """
        Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
        """
        if isinstance(inputs, (list, tuple)):
            for t in inputs:
                result = self._find_batch_size(t)
                if result is not None:
                    return result
        elif isinstance(inputs, Mapping):
            for key, value in inputs.items():
                result = self._find_batch_size(value)
                if result is not None:
                    return result
        elif isinstance(inputs, torch.Tensor):
            return inputs.shape[0] if len(inputs.shape) >= 1 else None
        elif isinstance(inputs, np.ndarray):
            return inputs.shape[0] if len(inputs.shape) >= 1 else None

    def _is_eval_epoch(self, cur_epoch):
        return (cur_epoch + 1 == self.cfg.SOLVER.MAX_EPOCH) or (cur_epoch + 1) % self.cfg.TRAIN.EVAL_PERIOD == 0

    @torch.no_grad()
    def eval_epoch(self):
        # set meters
        batch_time = AverageMeter('Time', ':6.3f') #!  여기 설명 필요함
        data_time = AverageMeter('Data', ':6.3f') #! 여기 설명 필요함
        metric_meters = self._get_metric_meters()
        loss_meters = self._get_loss_meters()
        progress = ProgressMeter(
            len(self.val_loader),
            [batch_time, data_time, *metric_meters, *loss_meters],
            prefix="Validation epoch[{}]".format(self.cur_epoch) #! 근데 왜 모든 epoch 안에 맨 마지막 epoch은 안나옴 150/151 이렇게 뜸. 원래 이런거면 +1 해줘야 되는거 아닌가?
            #! 맨 마지막 큰 epoch 도 29에서 끝남
        )
        log_dict = {}

        # switch to eval mode
        self.model.eval()

        start = time.time()
        for cur_iter, inputs in enumerate(self.val_loader):
            # measure data loading time
            data_time.update(time.time() - start)

            outputs = self.eval_step(inputs)

            # update metric and loss meters, and log to W&B
            batch_size = self._find_batch_size(inputs)
            self._update_metric_meters(metric_meters, outputs["metrics"], batch_size)
            self._update_loss_meters(loss_meters, outputs["losses"], batch_size)

            if self._is_display_iter(cur_iter):
                progress.display(cur_iter)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

        log_dict.update({
            f"Val/{metric_meter.name}": metric_meter.avg for metric_meter in metric_meters
        })
        log_dict.update({
            f"Val/{loss_meter.name}": loss_meter.avg for loss_meter in loss_meters
        })

        if self.cfg.WANDB.ENABLE:
            wandb.log(log_dict)

        # track the best model based on the first metric
        tracking_meter = metric_meters[0]

        return tracking_meter

    @torch.no_grad()
    def eval_step(self, inputs):
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        
        ground_truth = dec_window[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:].float()
        dec_zeros = torch.zeros_like(dec_window[:, -self.cfg.DATA.PRED_LEN:, :]).float()
        dec_window = torch.cat([dec_window[:, :self.cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float().cuda()
        
        # model_cfg = getattr(self.cfg.MODEL, self.cfg.MODEL_NAME.upper())
        model_cfg = self.cfg.MODEL
        
        if model_cfg.output_attention:
            pred, weights = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)[0]
        else:
            pred, weights = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)
        
        pred = pred[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:]
        
        loss = F.mse_loss(pred, ground_truth)
        metric = F.l1_loss(pred, ground_truth)
        
        outputs = dict(
            losses=(loss,),
            metrics=(metric,)
        )
        
        return outputs

    def _is_display_iter(self, cur_iter):
        return cur_iter % self.cfg.TRAIN.PRINT_FREQ == 0 or (cur_iter + 1) == len(self.val_loader)

    @torch.no_grad()
    def predict(self):
        self.load_best_model()

        # set to eval mode
        self.model.eval()

        # set meters
        metric_meters = self._get_metric_meters()
        loss_meters = self._get_loss_meters()
        progress = ProgressMeter(
            len(self.test_loader),
            [*metric_meters, *loss_meters],
            prefix="Test"
        )

        for cur_iter, inputs in enumerate(self.test_loader):
            inputs = prepare_inputs(inputs)
            outputs = self.model.get_anomaly_scores(inputs)

    def save_best_model(self):
        checkpoint = {
            "epoch": self.cur_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": self.cfg.dump(),
        }
        with open(mkdir(self.cfg.TRAIN.CHECKPOINT_DIR) / 'checkpoint_best.pth', "wb") as f:
            torch.save(checkpoint, f)

    def load_best_model(self):
        if self.cfg.TRAIN.ENABLE:
            model_path = os.path.join(self.cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")
        else:
            model_path = os.path.join(self.cfg.TEST.CHECKPOINT_DIR, "checkpoint_best.pth")
            
        if os.path.isfile(model_path):
            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")

            state_dict = checkpoint['model_state']
            msg = self.model.load_state_dict(state_dict, strict=True)
            assert set(msg.missing_keys) == set()

            ############################    MUC    ############################
            if not self.cfg.TRAIN.ENABLE:
                topk = self.cfg.TEST.TOPK
                for i in range(len(self.model.enc_embedding)):
                    if self.cfg.MODEL.softmax:
                        top_cells_indices = torch.topk(self.model.enc_embedding[i].weights, topk).indices
                    else:
                        top_cells_indices = torch.topk(torch.abs(self.model.enc_embedding[i].weights), topk).indices
                    
                    self.model.enc_embedding[i].cells = nn.ModuleList([self.model.enc_embedding[i].cells[j] for j in top_cells_indices])
                    self.model.enc_embedding[i].weights = nn.Parameter(torch.tensor([self.model.enc_embedding[i].weights[j] for j in top_cells_indices], device=self.model.enc_embedding[i].weights.device))
                    print(f"Layer {i} top cells: {top_cells_indices}")
            ############################    MUC    ############################
            
            print(f"Loaded pre-trained model from {model_path}")
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

        return self.model

def build_trainer(cfg, model):
    trainer = Trainer(cfg, model, cfg.MODEL.METRIC_NAMES, cfg.MODEL.LOSS_NAMES)
    return trainer

def prepare_inputs(inputs):
    # move data to the current GPU
    if isinstance(inputs, torch.Tensor):
        return inputs.float().cuda()
    elif isinstance(inputs, (tuple, list)):
        return type(inputs)(prepare_inputs(v) for v in inputs)

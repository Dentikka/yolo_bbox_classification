from comet_ml import Experiment

import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import Adam, SparseAdam, SGD, RAdam, NAdam, lr_scheduler
from torchvision.utils import make_grid
from torchvision import transforms


def get_experiment(cfg_exp: dict) -> Experiment:
    if cfg_exp is None:
        return None
    api_key_path = cfg_exp.pop('api_key_path')
    with open(api_key_path, 'r') as api_key_file:
        cfg_exp['api_key'] = api_key_file.readline()
    name = cfg_exp.pop('name')
    exp = Experiment(**cfg_exp)
    exp.set_name(name)
    return exp


def get_optimizer(parameters, cfg: dict):
    if cfg['type'] == 'adam':
        opt = Adam(parameters, lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0.0))
    elif cfg['type'] == 'radam':
        opt = RAdam(parameters, lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0.0))
    elif cfg['type'] == 'nadam':
        opt = NAdam(parameters, lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0.0), decoupled_weight_decay=True)
    elif cfg['type'] == 'sparse_adam':
        opt = SparseAdam(parameters, lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0.0))
    elif cfg['type'] == 'sgd':
        opt = SGD(parameters, lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0.0))
    else:
        raise NotImplementedError(f'Unknown optimizer in config: {cfg["type"]}')
    return opt


def get_scheduler(opt, lr_policy: dict):
    if len(lr_policy) == 0:
        return None
    if lr_policy['type'] == 'step':
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=lr_policy['step_size'],
            gamma=lr_policy['gamma'])
    elif lr_policy['type'] == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(
            opt,
            milestones=lr_policy['steps'],
            gamma=lr_policy['gamma'])
    elif lr_policy['type'] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=lr_policy['n_epochs']
        )
    else:
        raise NotImplementedError('Learning rate policy {} not implemented.'.
                                   format(lr_policy['type']))
    return scheduler


def log_images(experiment: Experiment, 
               name: str,
               epoch: int,
               batch_to_log: Tensor) -> None:
    inv_transform = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ],
                             std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                             std = [ 1., 1., 1. ]),
        transforms.ToPILImage(),
    ])
    grid = inv_transform(make_grid(batch_to_log, nrow=8, padding=2))
    experiment.log_image(grid, name=name, step=epoch)


def log_grads(experiment: Experiment, 
              epoch: int, 
              metrics_grad_log: dict) -> None:
    for key, value in metrics_grad_log.items():
        experiment.log_metric(key, torch.nanmean(torch.stack(value)), epoch=epoch, step=epoch)
    metrics_grad_log = defaultdict(list)
    return metrics_grad_log


def read_py_config(path: str) -> str:
    path = Path(path)
    sys.path.append(str(path.parent))
    line = f'import {path.stem} as cfg'
    return line


def bbox_xywh2xyxy(xc: float, yc: float, w: float, h: float) -> tuple[float]:
    xlt = xc - w/2
    xrb = xc + w/2
    ylt = yc - h/2
    yrb = yc + h/2
    return xlt, ylt, xrb, yrb


def bbox_xyxy2xywh(xlt: float, ylt: float, xrb: float, yrb: float) -> tuple[float]:
    xc = (xlt + xrb)/2
    yc = (ylt + yrb)/2
    w = xrb - xlt
    h = yrb - ylt
    return xc, yc, w, h

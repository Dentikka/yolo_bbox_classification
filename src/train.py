import os
from pathlib import Path

from collections import defaultdict

import comet_ml

import torch
from torch.cuda.amp import GradScaler

import argparse
from tqdm import tqdm

from dataset import get_dataset
from model import get_model
from loss import get_loss
from metrics import (
    compute_metrics,
    log_metrics,
    log_confusion_matrix,
)

from utils import (
    get_experiment,
    get_optimizer,
    get_scheduler,
    log_images,
    log_grads,
    read_py_config,
)

def train_epoch(model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                criterion,
                device,
                cfg):
    train_running_loss = []
    train_confidences = []
    train_predictions = []
    train_ground_truth = []

    model.train()

    if cfg.log_gradients:
        metrics_grad_log = defaultdict(list)
    
    pbar = tqdm(train_loader, leave=False, desc='Training')

    batch_to_log = None
    for img, target in pbar:
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.enable_mixed_presicion):
            preds = model(img)
            loss = criterion(preds, target)

        loss_item = loss.item()
        
        pbar.set_postfix_str(f'Loss: {loss_item:.4f}')

        train_running_loss.append(loss_item)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_ground_truth.extend(
            target
            .cpu()
            .numpy()
            .tolist()
        )
        train_confidences.extend(
            preds
            .softmax(dim=-1, dtype=torch.float32)
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        train_predictions.extend(
            preds
            .argmax(dim=-1)
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        if cfg.log_gradients:
            total_grad = 0
            for tag, value in model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_grad_log[f"Gradients/{tag}"].append(grad)
                    total_grad += grad

            metrics_grad_log["Gradients/Total"].append(total_grad)

        if batch_to_log is None:
            batch_to_log = img.to('cpu')

    if scheduler is not None:
        scheduler.step()

    results = {
        'running_loss': train_running_loss,
        'confidences': train_confidences,
        'predictions': train_predictions,
        'ground_truth': train_ground_truth,
        'images': batch_to_log
    }

    if cfg.log_gradients:
        results['metrics_grad_log'] = metrics_grad_log

    return results

@torch.no_grad()
def val_epoch(model,
              val_loader,
              criterion,
              device,
              cfg):
    
    val_confidences = []
    val_predictions = []
    val_ground_truth = []
    val_running_loss = []

    model.eval()

    batch_to_log = None
    for img, target in tqdm(val_loader, leave=False, desc='Evaluating'):
        img, target = img.to(device), target.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.enable_mixed_presicion):
            preds = model(img)
            loss = criterion(preds, target)

        loss_item = loss.item()

        val_running_loss.append(loss_item)
        
        val_ground_truth.extend(
            target
            .cpu()
            .numpy()
            .tolist()
        )
        val_confidences.extend(
            preds
            .softmax(dim=-1, dtype=torch.float32)
            .cpu()
            .numpy()
            .tolist()
        )
        val_predictions.extend(
            preds
            .argmax(dim=-1)
            .cpu()
            .numpy()
            .tolist()
        )

        if batch_to_log is None:
            batch_to_log = img.to('cpu')
    
    results = {
        'running_loss': val_running_loss,
        'confidences': val_confidences,
        'predictions': val_predictions,
        'ground_truth': val_ground_truth,
        'images': batch_to_log
    }

    return results


def train(model,
          train_loader,
          val_loader,
          optimizer,
          scheduler,
          criterion,
          experiment, 
          device,
          cfg):
    model_path = Path(cfg.model_save_path)
    model_path.mkdir(exist_ok=True, parents=True)
    n_epochs = cfg.n_epochs
    best_val_acc = 0.
    label_names = cfg.label_names

    scaler = GradScaler(enabled=cfg.enable_gradient_scaler)

    for epoch in tqdm(range(n_epochs), desc='Training epochs'):

        train_results = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
            device,
            cfg)

        val_results = val_epoch(
            model,
            val_loader,
            criterion,
            device,
            cfg)

        epoch_val_acc = None
        if experiment is not None:  # log metrics
            log_images(
                experiment,
                'Train', 
                epoch,
                train_results['images']
            )
            
            log_images(
                experiment,
                'Validation',
                epoch, 
                val_results['images']
            )

            train_metrics = compute_metrics(train_results, cfg.label_names)

            log_metrics(
                experiment,
                epoch,
                train_metrics,
                'Train'
            )

            val_metrics = compute_metrics(val_results, cfg.label_names)
            epoch_val_acc = val_metrics['acc']

            print('\nTrain:')
            for metric, value in train_metrics.items():
                print(f'\t{metric}: {value}')
            print('Validation:')
            for metric, value in val_metrics.items():
                print(f'\t{metric}: {value}')

            log_metrics(
                experiment,
                epoch,
                val_metrics,
                'Validation'
            )
            
            log_confusion_matrix(
                experiment,
                label_names,
                epoch,
                val_results,
                'Validation'
            )
            
            if cfg.log_gradients:
                log_grads(
                    experiment, 
                    epoch, 
                    train_results['metrics_grad_log']
                )

        if epoch_val_acc is not None:
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.state_dict(), Path(model_path, 'best.pth'))
        torch.save(model.state_dict(), Path(model_path, 'last.pth'))


def main():
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('-cfg', '--config', help='Config file path', type=str, default='', required=True)
    args = parser.parse_args()
    cfg_file = args.config
    exec(read_py_config(cfg_file), globals(), globals())
    train_loader = get_dataset(cfg.train_paths, cfg.train_data, cfg.train_pipeline)
    val_loader = get_dataset(cfg.val_paths, cfg.val_data, cfg.val_pipeline)
    label_names = cfg.label_names
    device = torch.device(cfg.device)
    model = get_model(cfg.model, len(label_names), device, compile=cfg.compile)
    model.set_backbone_state('freeze')
    optimizer = get_optimizer(model.parameters(), cfg=cfg.optimizer)
    scheduler = get_scheduler(optimizer, cfg.lr_policy)
    criterion = get_loss(cfg.criterion, cfg.device)
    experiment = get_experiment(cfg.experiment)
    experiment.log_code(cfg_file)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    experiment.log_code(os.path.join(dir_path, 'model.py'))
    train(model,
          train_loader,
          val_loader,
          optimizer,
          scheduler,
          criterion,
          experiment, 
          device,
          cfg)


if __name__ == '__main__':
    main()

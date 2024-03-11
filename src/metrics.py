import typing as tp
from comet_ml import Experiment

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(epoch_results: dict[str, tp.Any],
                    label_names: list[str] | None) -> dict[str, tp.Any]:
    running_loss = epoch_results['running_loss']
    confidences = epoch_results['confidences']
    predictions = epoch_results['predictions']
    ground_truth = epoch_results['ground_truth']
    classes = list(set(ground_truth))
    label_names = np.array(label_names)[classes]
    confidences = np.array(confidences)[:, classes]
    confidences /= confidences.sum(axis=1, keepdims=True)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    ground_truth = list(map(lambda cls: class_to_idx[cls], ground_truth))
    n_classes = len(classes)

    if n_classes > 2:
        classwise_precision = precision_score(ground_truth, predictions, average=None)
        classwise_recall = recall_score(ground_truth, predictions, average=None)
        classwise_roc_auc = roc_auc_score(ground_truth, confidences, average=None, multi_class='ovr')
        acc = accuracy_score(ground_truth, predictions)
        mean_precision = classwise_precision.mean()
        mean_recall = classwise_recall.mean()
        mean_roc_auc = classwise_roc_auc.mean()
        loss = np.mean(running_loss)
        metrics = {
            **{f'{classname}/precision': precision for classname, precision in zip(label_names, classwise_precision)},
            **{f'{classname}/recall': recall for classname, recall in zip(label_names, classwise_recall)},
            **{f'{classname}/roc_auc': roc_auc for classname, roc_auc in zip(label_names, classwise_roc_auc)},
            'acc': acc,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_roc_auc': mean_roc_auc,
            'loss': loss,
        }
    else:
        epoch_acc = accuracy_score(ground_truth, predictions)
        epoch_precision = precision_score(ground_truth, np.array(confidences).argmax(axis=1))
        epoch_recall = recall_score(ground_truth, np.array(confidences).argmax(axis=1))
        epoch_roc_auc = roc_auc_score(ground_truth, np.array(confidences)[:, 1])
        epoch_loss = np.mean(running_loss)
        metrics = {
            'acc': epoch_acc,
            'mean_precision': epoch_precision,
            'mean_recall': epoch_recall,
            'mean_roc_auc': epoch_roc_auc,
            'loss': epoch_loss,
        }
    return metrics


def log_metrics(experiment: Experiment,
                epoch: int,
                metrics: dict[str, tp.Any],
                fold: str='Train') -> None:
    for name, value in metrics.items():
        experiment.log_metric(f'{fold}/{name}', value, epoch=epoch, step=epoch)


def log_confusion_matrix(experiment: Experiment,
                         label_names: list[str],
                         epoch: int,
                         results: dict[str, tp.Any],
                         fold: str='Validation') -> None:
    experiment.log_confusion_matrix(results['ground_truth'],
                                    results['predictions'],
                                    labels=label_names, 
                                    title=f'{fold} confusion matrix',
                                    file_name=f'{fold}-confusion-matrix.json',
                                    epoch=epoch)

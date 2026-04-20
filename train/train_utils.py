# train_utils.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score


def focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1 - p)
    loss = alpha * (1 - pt).pow(gamma) * bce
    return loss.mean()


def cls_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rec_e = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_g = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return {"auc": auc, "f1": f1, "expert_recall": rec_e, "general_recall": rec_g}
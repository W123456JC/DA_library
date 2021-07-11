import torch
from torch.nn import functional as F


def cross_entropy(logits, labels, loss, reduction='mean'):
    if reduction == 'mean':
        return loss(logits, labels)
    elif reduction == 'sum':
        return loss.sum(logits, labels)
    else:
        raise ValueError



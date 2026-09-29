import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_class_priors(args):
    names = {
        "chexpert_plus": "hsa_her_chexpert_plus_base_probs.json",
        "iu_xray": "hsa_her_iu_base_probs.json",
        "mimic_cxr": "hsa_her_base_probs.json",
    }
    dataset = args.dataset
    if dataset not in names:
        raise ValueError(f"Unsupported dataset for disease priors: {dataset}")
    default_path = Path(__file__).resolve().parent.parent / "configs" / names[dataset]
    prior_path = Path(getattr(args, "base_probs_path", None) or default_path)
    with prior_path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    priors = torch.as_tensor(values, dtype=torch.float32)
    if priors.shape != (14,) or not torch.isfinite(priors).all() or not ((priors > 0) & (priors <= 1)).all():
        raise ValueError(f"Expected 14 disease prior probabilities in (0, 1]: {prior_path}")
    return priors


def adjust_logits(logits, priors):
    if logits.ndim != 2 or logits.shape[1] != 14:
        raise ValueError(f"Expected disease logits [batch, 14], got {tuple(logits.shape)}")
    return logits - priors.to(device=logits.device, dtype=logits.dtype).log().unsqueeze(0)


def binary_targets(labels):
    if labels.ndim != 2 or labels.shape[1] != 14:
        raise ValueError(f"Expected CheXbert labels [batch, 14], got {tuple(labels.shape)}")
    if not torch.isin(labels, torch.tensor([0, 1, 2, 3], device=labels.device)).all():
        raise ValueError("CheXbert labels must use 0=blank, 1=positive, 2=negative, 3=uncertain")
    valid = (labels == 1) | (labels == 2)
    target = (labels == 1).to(torch.float32)
    return target, valid


def binary_classification_loss(adjusted_logits, labels):
    target, valid = binary_targets(labels)
    if not valid.any():
        return adjusted_logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(adjusted_logits, target.to(adjusted_logits.dtype), reduction="none")
    return losses[valid].mean()


def binary_classification_accuracy(adjusted_logits, labels):
    target, valid = binary_targets(labels)
    if not valid.any():
        return adjusted_logits.new_zeros(())
    predicted = adjusted_logits.sigmoid() >= 0.5
    return (predicted[valid] == target[valid].bool()).float().mean()

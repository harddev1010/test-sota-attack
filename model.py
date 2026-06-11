"""
model.py — load the frozen EfficientNetV2-L classifier and expose a simple predict().

Key idea: attacks (foolbox/torchattacks) like to work in [0,1] pixel space, but the
network expects MEAN/STD-normalized inputs. We hide that mismatch inside a tiny wrapper
module so every caller can just hand us [0,1] images and forget about normalization.

NOTE: the label logic here (argmax over softmax -> CATEGORIES name) must match
perturbnet/model.py. If the real validator does something different (e.g. a custom
head, top-k, or a different label list), mirror that change here.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_l

from config import WEIGHTS, CATEGORIES, MEAN_T, STD_T, DEVICE, normalize


class NormalizedModel(nn.Module):
    """Wraps the classifier so it accepts images in [0,1] and normalizes internally."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        # register_buffer = "part of the model state, but NOT a trainable parameter".
        # This makes mean/std move with .to(device) automatically.
        self.register_buffer("mean", MEAN_T.clone())
        self.register_buffer("std", STD_T.clone())

    def forward(self, x):
        # x is a batch of images in [0,1]; normalize then run the real network.
        return self.base((x - self.mean) / self.std)


def load_model():
    """Build, freeze, and return the wrapped model ready for inference/attacks."""
    base = efficientnet_v2_l(weights=WEIGHTS)   # download/cache the official weights
    base.eval()                                 # eval mode: no dropout, frozen BatchNorm stats
    for p in base.parameters():
        p.requires_grad_(False)                 # freeze: we attack the INPUT, never the weights
    model = NormalizedModel(base).to(DEVICE).eval()
    return model


def predict(model, x):
    """
    Classify a SINGLE image tensor in [0,1] space.

    Returns (label_string, confidence_float).
    Accepts shape (3,H,W) or (1,3,H,W).
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)                      # add the batch dimension -> (1,3,H,W)
    with torch.no_grad():                       # inference only: skip gradient bookkeeping
        logits = model(x.to(DEVICE))            # raw scores per class
        probs = logits.softmax(dim=1)           # turn scores into probabilities
        conf, idx = probs.max(dim=1)            # best class + its probability
    return CATEGORIES[idx.item()], conf.item()  # map index -> human-readable ImageNet label


def predict_index(model, x):
    """Same as predict() but returns the integer class index (attacks need the index)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    with torch.no_grad():
        return model(x.to(DEVICE)).argmax(dim=1).item()

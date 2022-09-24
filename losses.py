import numpy as np
import torch
import torch.nn as nn


def all_vs_QCD(output, target):
    mask_bb = (target[:,0] == 1) | (target[:,-1] == 1)
    mask_cc = (target[:,1] == 1) | (target[:,-1] == 1)
    mask_qq = (target[:,2] == 1) | (target[:,-1] == 1)
    loss = nn.functional.binary_cross_entropy(output[mask_bb], target[mask_bb]) + nn.functional.binary_cross_entropy(output[mask_cc], target[mask_cc]) + nn.functional.binary_cross_entropy(output[mask_qq], target[mask_qq]) 
    return loss

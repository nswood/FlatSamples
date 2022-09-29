import numpy as np
import torch
import torch.nn as nn


def adversarial(output, target, full_labels, one_hots, lambda_adversarial):
    #https://stackoverflow.com/questions/71049941/create-one-hot-encoding-for-values-of-histogram-bins
    #hist = torch.histc(mass, bins=20, min=30., max=400.).cuda()
    #hist = hist.to(torch.int32)
    #bins = torch.linspace(40.,350.,20)
    #one_hots = torch.eye(len(hist)).cuda()
    #one_hots = torch.repeat_interleave(one_hots, hist, dim=0)
    
    #mass histogram for true b events weighted by b prob 

    #print(one_hots.shape)
    #torch.mm(torch.diag(target[:,0]),output).shape
    #mass histogram for true b events
    hist_alltag_b = torch.mm(torch.transpose(one_hots,0,1), torch.mm(torch.diag(full_labels[:,0]),output))
    #mass histogram for true b events weighted by qcd prob
    hist_qcdtag_b = hist_alltag_b[:,-1]/torch.sum(hist_alltag_b[:,-1])
    #mass histogram for true b events weighted by b prob
    hist_btag_b   = hist_alltag_b[:,0]/torch.sum(hist_alltag_b[:,0])

    hist_average_b = (hist_btag_b + hist_qcdtag_b)/2.0
    #print(hist_average_b)
    bce_loss = all_vs_QCD(output, target)
    
    return bce_loss + lambda_adversarial*(torch.nn.functional.kl_div(hist_qcdtag_b,hist_average_b) + torch.nn.functional.kl_div(hist_btag_b,hist_average_b))/2.
def all_vs_QCD(output, target):


    mask_bb = (target[:,0] == 1) | (target[:,-1] == 1)
    mask_cc = (target[:,1] == 1) | (target[:,-1] == 1)
    mask_qq = (target[:,2] == 1) | (target[:,-1] == 1) 
    #print(torch.where(mask_bb == True))
    #print(torch.where(mask_cc == True))
    #print(torch.where(mask_qq == True))
    #print(nn.functional.binary_cross_entropy(output[mask_bb], target[mask_bb]))
    #print(nn.functional.binary_cross_entropy(output[mask_cc], target[mask_cc]))
    #print(nn.functional.binary_cross_entropy(output[mask_qq], target[mask_qq]))
    loss = nn.functional.binary_cross_entropy(output[mask_bb], target[mask_bb]) + nn.functional.binary_cross_entropy(output[mask_cc], target[mask_cc]) + nn.functional.binary_cross_entropy(output[mask_qq], target[mask_qq]) 
    return loss

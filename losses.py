import numpy as np
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DiscoCorr(nn.Module):
    def __init__(self,background_only=False,anti=False,background_label=1,power=2):
        self.backonly = background_only
        self.background_label = background_label
        self.power = power
        self.anti = anti

    def distance_corr(self,var_1,var_2,normedweight,power=1):
        xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
        yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
        amat = (xx-yy).abs()
        del xx,yy

        amatavg = torch.mean(amat*normedweight,dim=1)
        Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))-amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))+torch.mean(amatavg*normedweight)
        del amat

        xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
        yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
        bmat = (xx-yy).abs()
        del xx,yy

        bmatavg = torch.mean(bmat*normedweight,dim=1)
        Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
          -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
          +torch.mean(bmatavg*normedweight)
        del bmat

        ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
        AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
        BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)
        del Bmat, Amat

        if(power==1):
            dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
        elif(power==2):
            dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
        else:
            dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power

        return dCorr

    def __call__(self,pred,x_biased,weights=None):
        xweights = torch.ones_like(pred)
        disco = self.distance_corr(x_biased,pred,normedweight=xweights,power=self.power)
        if self.anti:
            disco = 1-disco
        return disco

def disco(output, target, mass, LAMBDA_ADV=10.):
    disco = DiscoCorr()
    bce_loss = nn.functional.binary_cross_entropy(output,target)
    #print("output",output.shape, "mass",mass.shape)
    return bce_loss \
        + LAMBDA_ADV*(disco(output[:,0], mass)) \

def jsd(output, target, one_hots, n_massbins=20,LAMBDA_ADV=10.):
    #https://stackoverflow.com/questions/71049941/create-one-hot-encoding-for-values-of-histogram-bins
    one_hots = nn.functional.one_hot(one_hots, num_classes=-1).to(torch.float32)



    #mass histogram for true b events weighted by b,qcd prob 
    hist_alltag_b = torch.mm(torch.transpose(one_hots,0,1), torch.mm(torch.diag(torch.where(target[:,0]==True,1.0,0.0)),output))
    #mass histogram for true b events weighted by qcd prob
    hist_qcdtag_b = hist_alltag_b[:,-1]/torch.sum(hist_alltag_b[:,-1],axis=0)
    #mass histogram for true b events weighted by b prob
    hist_btag_b   = hist_alltag_b[:,0]/torch.sum(hist_alltag_b[:,0],axis=0)
    #average of true b histogram
    hist_average_b = (hist_btag_b + hist_qcdtag_b)/2.0

    #mass histogram for true qcd events weighted by b,qcd prob 
    hist_alltag_qcd = torch.mm(torch.transpose(one_hots,0,1), torch.mm(torch.diag(torch.where(target[:,-1]==True,1.0,0.0)),output))
    #mass histogram for true qcd events weighted by qcd prob 
    hist_qcdtag_qcd = hist_alltag_qcd[:,-1]/torch.sum(hist_alltag_qcd[:,-1],axis=0)
    #mass histogram for true qcd events weighted by b prob 
    hist_btag_qcd = hist_alltag_qcd[:,0]/torch.sum(hist_alltag_qcd[:,0],axis=0)
    #average of true qcd histogram
    hist_average_qcd = (hist_btag_qcd + hist_qcdtag_qcd)/2.0
    bce_loss = nn.functional.binary_cross_entropy(output,target)
    
    return bce_loss \
         + LAMBDA_ADV*(torch.nn.functional.kl_div(hist_qcdtag_b,hist_average_b) + torch.nn.functional.kl_div(hist_btag_b,hist_average_b))/2.\
         + LAMBDA_ADV*(torch.nn.functional.kl_div(hist_btag_qcd,hist_average_qcd) + torch.nn.functional.kl_div(hist_qcdtag_qcd,hist_average_qcd))/2.\

def all_vs_QCD(output, target):


    mask_bb = (target[:,0] == 1) | (target[:,-1] == 1)
    mask_cc = (target[:,1] == 1) | (target[:,-1] == 1)
    mask_qq = (target[:,2] == 1) | (target[:,-1] == 1) 

    loss = nn.functional.binary_cross_entropy(output[mask_bb], target[mask_bb]) + nn.functional.binary_cross_entropy(output[mask_cc], target[mask_cc]) + nn.functional.binary_cross_entropy(output[mask_qq], target[mask_qq]) 
    return loss

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:26:50 2019

@author: marry
@target: Losses
"""

import torch.nn as nn
import torch
import numpy as np


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.partial_loss = nn.L1Loss()
        
    def __call__(self, F1, F2):
        loss = 0
        L1 = len(F1)
        L2 = len(F2)
        if L1 != L2:
            raise Exception("Unmatch input features")
        for i in range(L1):
            loss += self.partial_loss(F1[i], F2[i])
        return loss
    
    
class AdversialLoss(nn.Module):
    def __init__(self):
        super(AdversialLoss, self).__init__()
        self.register_buffer('true_label', torch.tensor(1.0))
        self.register_buffer('false_label', torch.tensor(0.0))
        
        self.loss = nn.MSELoss()
    
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real is True:
            target_tensor = self.true_label
        else:
            target_tensor = self.false_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        """
        prediction -tensor
        target_is_real -bool
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss
    

class PerceptualLoss(nn.Module):
    def __init__(self):
        self.criterion = nn.MSELoss()
    
    def __call__(self, vgg, I_pred, I_tar):
        loss = 0.0
        Fs_pred = vgg(I_pred)
        Fs_tar = vgg(I_tar)
        if type(Fs_pred) == list:
            L = len(Fs_pred)
            for i in range(L):
                loss += self.criterion(Fs_pred[i], Fs_tar[i])
        else:
            loss = self.criterion(Fs_pred, Fs_tar)
            
        return loss
    
    
class PerceptualL1Loss(nn.Module):
    def __init__(self):
        self.criterion = nn.L1Loss()
    
    def __call__(self, y_pred, y_label):
        loss = 0.0
        Fs_pred = vgg(I_pred)
        Fs_tar = vgg(I_tar)
        if type(Fs_pred) == list:
            L = len(Fs_pred)
            for i in range(L):
                loss += self.criterion(Fs_pred[i], Fs_tar[i])
        else:
            loss = self.criterion(Fs_pred, Fs_tar)
            
        return loss

class Wasserstein_loss(nn.Module):
    def __init__(self):
        pass
    
    def __call__(self, y_pred, y_label):
        return (y_pred * y_label).mean()
    
    
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        self.eps = 1e-12
    
    def __call__(self, y_pred, y_label):
        y_pred = np.clip(y_pred, self.eps, 1.0-self.eps)
        return -(y_label * np.log(y_pred+self.eps) + (1-y_label) * (1-y_pred+self.eps).log()).mean()

class BinaryAccuracyMetric(nn.Module):
    def __init__(self):
        pass
    
    def __call__(self, y_pred, y_label):
        y_pred = np.where(y_pred > 0.5, 1, 0)
        y_pred = y_pred.astype(np.int32)
        return (np.equal(y_label, y_pred)).mean()
    

   



   




"""test"""
# =============================================================================
# from torch.autograd import Variable
# 
# A = Variable(torch.FloatTensor([1, 2, 3]))
# AdvLoss = AdversialLoss()
# print(AdvLoss(A, True))
# =============================================================================

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:52:13 2019

@author: marry
@target: Generator --CartoonGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNormalization(nn.Module):
    def __init__(self, dim, eps=1e-9):
        '''
        dim -int -[1]
        '''
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))   
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        # x' = sclae* * x + shift
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        # initialization or reseting
        self.scale.data.uniform_()
        self.shift.data.zero_()
        
    def __call__(self, x):
        '''
        x --[N, C, H, W]
        mean, var, scale_broadcast, shift_broadcast --[N, C, H, W]
        '''
        n = x.size(2)*x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        # [N, C, H, W] -> [N, C, HW]
        '''unsqueeze(i) where i means we will expand the dim as the i-th dim'''
        
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # [N, C] -> [H, C, 1] -> [N, C, 1, 1] -> [N, C, H, W]
        # the avg is based on each channel crossing the 'H-W' plane
        
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        # [dim] -> [dim, 1] -> [dim, 1, 1] -> [1, dim, 1, 1] -> [N, dim, H, W]
        # for each channel
            
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
            
    
"""
torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, bias=True)
torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                         stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
"""        
    
class CartoonGenerator(nn.Module):
    def __init__(self):
        super(CartoonGenerator, self).__init__()
        
        """Down Convolution"""
        self.refpad0_1_1 = nn.ReflectionPad2d(3)
        self.conv0_1_1 = nn.Conv2d(3, 64, 7)
        self.in0_1_1 = InstanceNormalization(64)
        # relu
        # [H, W]
        
        self.conv0_2_1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv0_2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.in0_2_1 = InstanceNormalization(128)
        # relu
        # [H/2, W/2]
    
        self.conv0_3_1 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv0_3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.in0_3_1 = InstanceNormalization(256)
        # relu
        # [H/4, W/4]
        
        """Residual Blocks"""
        self.refpad0_4_1 = nn.ReflectionPad2d(1)
        self.conv0_4_1 = nn.Conv2d(256, 256, 3)
        self.in0_4_1 = InstanceNormalization(256)
        # relu
        self.refpad0_4_2 = nn.ReflectionPad2d(1)
        self.conv0_4_2 = nn.Conv2d(256, 256, 3)
        self.in0_4_2 = InstanceNormalization(256)
        # + input
        
        self.refpad0_5_1 = nn.ReflectionPad2d(1)
        self.conv0_5_1 = nn.Conv2d(256, 256, 3)
        self.in0_5_1 = InstanceNormalization(256)
        # relu
        self.refpad0_5_2 = nn.ReflectionPad2d(1)
        self.conv0_5_2 = nn.Conv2d(256, 256, 3)
        self.in0_5_2 = InstanceNormalization(256)
        # + input
        
        self.refpad0_6_1 = nn.ReflectionPad2d(1)
        self.conv0_6_1 = nn.Conv2d(256, 256, 3)
        self.in0_6_1 = InstanceNormalization(256)
        # relu
        self.refpad0_6_2 = nn.ReflectionPad2d(1)
        self.conv0_6_2 = nn.Conv2d(256, 256, 3)
        self.in0_6_2 = InstanceNormalization(256)
        # + input
        
        self.refpad0_7_1 = nn.ReflectionPad2d(1)
        self.conv0_7_1 = nn.Conv2d(256, 256, 3)
        self.in0_7_1 = InstanceNormalization(256)
        # relu
        self.refpad0_7_2 = nn.ReflectionPad2d(1)
        self.conv0_7_2 = nn.Conv2d(256, 256, 3)
        self.in0_7_2 = InstanceNormalization(256)
        # + input
        
        self.refpad0_8_1 = nn.ReflectionPad2d(1)
        self.conv0_8_1 = nn.Conv2d(256, 256, 3)
        self.in0_8_1 = InstanceNormalization(256)
        # relu
        self.refpad0_8_2 = nn.ReflectionPad2d(1)
        self.conv0_8_2 = nn.Conv2d(256, 256, 3)
        self.in0_8_2 = InstanceNormalization(256)
        # + input
        
        self.refpad0_9_1 = nn.ReflectionPad2d(1)
        self.conv0_9_1 = nn.Conv2d(256, 256, 3)
        self.in0_9_1 = InstanceNormalization(256)
        # relu
        self.refpad0_9_2 = nn.ReflectionPad2d(1)
        self.conv0_9_2 = nn.Conv2d(256, 256, 3)
        self.in0_9_2 = InstanceNormalization(256)
        # + input
        
        self.refpad0_10_1 = nn.ReflectionPad2d(1)
        self.conv0_10_1 = nn.Conv2d(256, 256, 3)
        self.in0_10_1 = InstanceNormalization(256)
        # relu
        self.refpad0_10_2 = nn.ReflectionPad2d(1)
        self.conv0_10_2 = nn.Conv2d(256, 256, 3)
        self.in0_10_2 = InstanceNormalization(256)
        # + input
        
        self.refpad0_11_1 = nn.ReflectionPad2d(1)
        self.conv0_11_1 = nn.Conv2d(256, 256, 3)
        self.in0_11_1 = InstanceNormalization(256)
        # relu
        self.refpad0_11_2 = nn.ReflectionPad2d(1)
        self.conv0_11_2 = nn.Conv2d(256, 256, 3)
        self.in0_11_2 = InstanceNormalization(256)
        # + input
        
        """UP Deconvolution"""
        self.deconv0_12_1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv0_12_2 = nn.ConvTranspose2d(128, 128, 3, 1, 1)
        self.in0_12_1 = InstanceNormalization(128)
        # relu
        
        self.deconv0_13_1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv0_13_2 = nn.ConvTranspose2d(64, 64, 3, 1, 1)
        self.in0_13_1 = InstanceNormalization(64)
        # relu
        
        self.refpad0_14_1 = nn.ReflectionPad2d(3)
        self.deconv0_14_1 = nn.Conv2d(64, 3, 7)
        # tanh
        
    def forward(self, x):
        y = F.relu(self.in0_1_1(self.conv0_1_1(self.refpad0_1_1(x))))
        y = F.relu(self.in0_2_1(self.conv0_2_2(self.conv0_2_1(y))))
        t04 = F.relu(self.in0_3_1(self.conv0_3_2(self.conv0_3_1(y))))
        
        """"""
        y = F.relu(self.in0_4_1(self.conv0_4_1(self.refpad0_4_1(t04))))
        t05 = self.in0_4_2(self.conv0_4_2(self.refpad0_4_2(y))) + t04
        
        y = F.relu(self.in0_5_1(self.conv0_5_1(self.refpad0_5_1(t05))))
        t06 = self.in0_5_2(self.conv0_5_2(self.refpad0_5_2(y))) + t05
        
        y = F.relu(self.in0_6_1(self.conv0_6_1(self.refpad0_6_1(t06))))
        t07 = self.in0_6_2(self.conv0_6_2(self.refpad0_6_2(y))) + t06
        
        y = F.relu(self.in0_7_1(self.conv0_7_1(self.refpad0_7_1(t07))))
        t08 = self.in0_7_2(self.conv0_7_2(self.refpad0_7_2(y))) + t07
        
        y = F.relu(self.in0_8_1(self.conv0_8_1(self.refpad0_8_1(t08))))
        t09 = self.in0_8_2(self.conv0_8_2(self.refpad0_8_2(y))) + t08
        
        y = F.relu(self.in0_9_1(self.conv0_9_1(self.refpad0_9_1(t09))))
        t10 = self.in0_9_2(self.conv0_9_2(self.refpad0_9_2(y))) + t09
        
        y = F.relu(self.in0_10_1(self.conv0_10_1(self.refpad0_10_1(t10))))
        t11 = self.in0_10_2(self.conv0_10_2(self.refpad0_10_2(y))) + t10
        
        y = F.relu(self.in0_11_1(self.conv0_11_1(self.refpad0_11_1(t11))))
        y = self.in0_11_2(self.conv0_11_2(self.refpad0_11_2(y))) + t11
        """"""
        
        y = F.relu(self.in0_12_1(self.deconv0_12_2(self.deconv0_12_1(y))))
        y = F.relu(self.in0_13_1(self.deconv0_13_2(self.deconv0_13_1(y))))
        y = torch.tanh(self.deconv0_14_1(self.refpad0_14_1(y)))

        return y
    
    
"""Testing"""
'''
from torch.autograd import Variable

G = CartoonGenerator().cuda()
input = torch.FloatTensor(1,3, 256, 256)
input = Variable(input).cuda()
output = G(input)
    
print(output.data.size())
# [N, 3, 256, 256]
'''        
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:17:11 2019

@author: marry
@target: Descriminator --CartoonGAN
"""


import torch.nn as nn
import torch.nn.functional as F
from Generator import InstanceNormalization

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, 3, 1)
        # leak_relu
        
        self.conv_2_1 = nn.Conv2d(32, 64, 3, 2, 1)
        # leak_relu
        self.conv_2_2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.in_2 = InstanceNormalization(128)
        # leak_relu
        
        self.conv_3_1 = nn.Conv2d(128, 128, 3, 2, 1)
        # leak_relu
        self.conv_3_2 = nn.Conv2d(128, 256, 3, 1, 1)
        self.in_3 = InstanceNormalization(256)
        # leak_relu
        
        self.conv_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.in_4 = InstanceNormalization(256)
        # leak_relu
        
        self.conv5 = nn.Conv2d(256, 1, 3, 1, 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv_1(x), negative_slope=0.2)
        
        x = F.leaky_relu(self.conv_2_1(x), negative_slope=0.2)
        x = F.leaky_relu(self.in_2(self.conv_2_2(x)), negative_slope=0.2)
        
        x = F.leaky_relu(self.conv_3_1(x), negative_slope=0.2)
        x = F.leaky_relu(self.in_3(self.conv_3_2(x)), negative_slope=0.2)
        
        x = F.leaky_relu(self.in_4(self.conv_4(x)), negative_slope=0.2)
        
        x = self.conv5(x)
        
        return x
    
"""
from torch.autograd import Variable

D = Discriminator().cuda()
input = torch.FloatTensor(1,3, 256, 256)
input = Variable(input).cuda()
output = D(input)
    
print(output.data.size())
# torch.Size([1, 1, 64, 64])
"""
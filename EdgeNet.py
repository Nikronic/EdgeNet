#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[3]:


class CBR(nn.Module):
    def __init__(self, input_channel, output_channel):
        """
        It consists of the 5x5 convolutions with stride=1, padding=2, and a batch normalization, followed by
        a rectified linear unit (ReLU)
        :param input_channel: input channel size
        :param output_channel: output channel size
        """
        assert (input_channel > 0 and output_channel > 0)

        super(CBR, self).__init__()
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size=5, stride=1, padding=2),
                  nn.BatchNorm2d(num_features=output_channel), nn.ReLU(0.2),
                  nn.ReLU(0.2)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# In[4]:


class C(nn.Module):
    def __init__(self, input_channel, output_channel):
        """
        At the final layer, a 3x3 convolution is used to map each 64-component feature vector to the desired
        number of classes.
        :param input_channel: input channel size
        :param output_channel: output channel size
        """
        super(C, self).__init__()
        self.layer = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.layer(x)


# In[9]:


class EdgeNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        """
        Implementation of CoarseNet, a modified version of UNet.
        (https://arxiv.org/abs/1505.04597 - Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015))
        :param input_channels: number of input channels of input images to network.
        :param output_channels: number of output channels of output images of network.
        :param depth: depth of network
        :param filters: number of filters in each layer (Each layer x2 the value).
        """

        super(EdgeNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        
        self.cbr0 = CBR(input_channels, 32)
        self.cbr1 = CBR(32, 32)
        self.cbr2 = CBR(32, 32)
        self.cbr3 = CBR(32, 32)
        self.cbr4 = CBR(32, 32)

        # final
        self.final = C(32, self.output_channels)

    def forward(self, x):
        c1 = self.cbr0(x)  # 3>32
        c1 = self.cbr1(c1)  # 32>32
        c1 = self.cbr2(c1)  # 32>32
        c1 = self.cbr3(c1)  # 32>32
        c1 = self.cbr4(c1)  # 32>32
        
        c1 = self.final(c1)
         
        


# In[10]:


x = torch.randn(1, 3, 256, 256)

model = CBR()
x = CBR(x)


# In[13]:


type(x)


# In[ ]:





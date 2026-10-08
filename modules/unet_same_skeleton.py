# Copyright (c) 2025 Yaolei Qi
# Copyright (c) 2026 Heng Zhang
# SPDX-License-Identifier: MIT
# Adapted from DSCNet; see THIRD_PARTY_NOTICES.md.
"""Standard branches only; same widths, depth, GN/ReLU and bilinear skips."""
import torch
from torch import nn
from S3_DSCNet import EncoderConv,DecoderConv
class UNetSameSkeleton(nn.Module):
    def __init__(self):
        super().__init__();self.number=16
        specs=[('conv00',1,16,EncoderConv),('conv1',16,16,EncoderConv),
          ('conv20',16,32,EncoderConv),('conv3',32,32,EncoderConv),
          ('conv40',32,64,EncoderConv),('conv5',64,64,EncoderConv),
          ('conv60',64,128,EncoderConv),('conv7',128,128,EncoderConv),
          ('conv120',192,64,EncoderConv),('conv13',64,64,EncoderConv),
          ('conv140',96,32,DecoderConv),('conv15',32,32,DecoderConv),
          ('conv160',48,16,DecoderConv),('conv17',16,16,DecoderConv)]
        for name,i,o,cls in specs:setattr(self,name,cls(i,o))
        self.out_conv=nn.Conv2d(16,1,1);self.maxpooling=nn.MaxPool2d(2)
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True);self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        a=self.conv1(self.conv00(x));b=self.conv3(self.conv20(self.maxpooling(a)))
        c=self.conv5(self.conv40(self.maxpooling(b)));d=self.conv7(self.conv60(self.maxpooling(c)))
        c2=self.conv13(self.conv120(torch.cat([self.up(d),c],1)))
        b2=self.conv15(self.conv140(torch.cat([self.up(c2),b],1)))
        a2=self.conv17(self.conv160(torch.cat([self.up(b2),a],1)))
        return self.sigmoid(self.out_conv(a2))

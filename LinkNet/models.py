from __future__ import print_function

import math
import os
import random

import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from SPIN import spin

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, group=1):
        super(DecoderBlock, self).__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1, groups=group)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=group,
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1, groups=group)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class LinkNet34(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, pretrainresnet=False):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained = pretrainresnet)

        if in_channels==3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(in_channels, filters[0], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
        self.return_features = False
        self.tanh = nn.Tanh()

        for m in [self.finaldeconv1, self.finalconv2]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rows = x.size()[2]
        cols = x.size()[3]
        
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = (
            self.decoder4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
            ]
            + e3
        )
        d3 = (
            self.decoder3(d4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
            ]
            + e2
        )
        d2 = (
            self.decoder2(d3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
            ]
            + e1
        )
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5


class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x))
        dilate2_out = F.relu(self.dilate2(dilate1_out))
        dilate3_out = F.relu(self.dilate3(dilate2_out))
        dilate4_out = F.relu(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out
    
class DLinkNet34(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, pretrainresnet=False):
        super(DLinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained = pretrainresnet)

        if in_channels==3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(in_channels, filters[0], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Center Dblock
        self.dblock = Dblock(512)

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
        self.return_features = False
        self.tanh = nn.Tanh()

        for m in [self.finaldeconv1, self.finalconv2]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rows = x.size()[2]
        cols = x.size()[3]
        
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder with Skip Connections
        d4 = (
            self.decoder4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
            ]
            + e3
        )
        d3 = (
            self.decoder3(d4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
            ]
            + e2
        )
        d2 = (
            self.decoder2(d3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
            ]
            + e1
        )
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5

class DLinkNet34MTL(nn.Module):
    def __init__(self, task1_classes=2, task2_classes=37, pretrainresnet=False):
        super(DLinkNet34MTL, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained = pretrainresnet)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Center Dblock
        self.dblock = Dblock(512)

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv3 = nn.Conv2d(32, task1_classes, 2, padding=1)

        # Decoder
        self.a_decoder4 = DecoderBlock(filters[3], filters[2])
        self.a_decoder3 = DecoderBlock(filters[2], filters[1])
        self.a_decoder2 = DecoderBlock(filters[1], filters[0])
        self.a_decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.a_finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.a_finalrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.a_finalconv2 = nn.Conv2d(32, 32, 3)
        self.a_finalrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.a_finalconv3 = nn.Conv2d(32, task2_classes, 2, padding=1)

        for m in [
            self.finaldeconv1,
            self.finalconv2,
            self.a_finaldeconv1,
            self.a_finalconv2,
        ]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rows = x.size()[2]
        cols = x.size()[3]

        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center Dblock
        e4 = self.dblock(e4)

        # Decoder with Skip Connections
        d4 = (
            self.decoder4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
            ]
            + e3
        )
        d3 = (
            self.decoder3(d4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
            ]
            + e2
        )
        d2 = (
            self.decoder2(d3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
            ]
            + e1
        )
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        # Decoder with Skip Connections
        a_d4 = (
            self.a_decoder4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
            ]
            + e3
        )
        a_d3 = (
            self.a_decoder3(a_d4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
            ]
            + e2
        )
        a_d2 = (
            self.a_decoder2(a_d3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
            ]
            + e1
        )
        a_d1 = self.a_decoder1(a_d2)

        # Final Classification
        a_f1 = self.a_finaldeconv1(a_d1)
        a_f2 = self.a_finalrelu1(a_f1)
        a_f3 = self.a_finalconv2(a_f2)
        a_f4 = self.a_finalrelu2(a_f3)
        a_f5 = self.a_finalconv3(a_f4)

        return f5, a_f5
    

class SpinDLinkNet34MTL(nn.Module):
    def __init__(self, task1_classes=2, task2_classes=37, pretrainresnet=False):
        super(SpinDLinkNet34MTL, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained = pretrainresnet)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Center Dblock
        self.dblock = Dblock(512)

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv3 = nn.Conv2d(32, task1_classes, 2, padding=1)

        # Decoder
        self.a_decoder4 = DecoderBlock(filters[3], filters[2])
        self.a_decoder3 = DecoderBlock(filters[2], filters[1])
        self.a_decoder2 = DecoderBlock(filters[1], filters[0])
        self.a_decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.a_finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.a_finalrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.a_finalconv2 = nn.Conv2d(32, 32, 3)
        self.a_finalrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.a_finalconv3 = nn.Conv2d(32, task2_classes, 2, padding=1)

        #Spin module
        self.dgcn_seg_1 = spin(planes=32, ratio=1)
        self.dgcn_seg_2 = spin(planes=32, ratio=1)

        for m in [
            self.finaldeconv1,
            self.finalconv2,
            self.a_finaldeconv1,
            self.a_finalconv2,
        ]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rows = x.size()[2]
        cols = x.size()[3]

        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center Dblock
        e4 = self.dblock(e4)

        # Decoder with Skip Connections
        d4 = (
            self.decoder4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
            ]
            + e3
        )
        d3 = (
            self.decoder3(d4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
            ]
            + e2
        )
        d2 = (
            self.decoder2(d3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
            ]
            + e1
        )
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f2 = self.dgcn_seg_1(f2) #Graph reasoning at LEVEL 4 - 257 x 257 - SPIN layer 1
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f4 = self.dgcn_seg_2(f4) #Graph reasoning at LEVEL 4 - 255 x 255 - SPIN layer 2
        f5 = self.finalconv3(f4)

        # Decoder with Skip Connections
        a_d4 = (
            self.a_decoder4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
            ]
            + e3
        )
        a_d3 = (
            self.a_decoder3(a_d4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
            ]
            + e2
        )
        a_d2 = (
            self.a_decoder2(a_d3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
            ]
            + e1
        )
        a_d1 = self.a_decoder1(a_d2)

        # Final Classification
        a_f1 = self.a_finaldeconv1(a_d1)
        a_f2 = self.a_finalrelu1(a_f1)
        a_f3 = self.a_finalconv2(a_f2)
        a_f4 = self.a_finalrelu2(a_f3)
        a_f5 = self.a_finalconv3(a_f4)

        return f5, a_f5


MODELS = {"LinkNet": LinkNet34, 
          "DLinkNet": DLinkNet34, 
          "DLinkNetMTL": DLinkNet34MTL, 
          "SpinDLinkNetMTL": SpinDLinkNet34MTL}
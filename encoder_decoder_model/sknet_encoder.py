from __future__ import division
""" 
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollar, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
import from https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
"""
from collections import OrderedDict
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

#__all__ = ['resnext50', 'resnext101', 'resnext152']
__all__ = ['sknet50', 'sknet101', 'sknet152']

class Bottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(Bottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(D*C)

        self.conv2_d1 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2_d1   = nn.BatchNorm2d(D*C)
        self.conv2_d2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=2, groups=C, bias=False, dilation=2)
        self.bn2_d2   = nn.BatchNorm2d(D*C)

        down_scale = 16.
        dim        = int(round(D*C / down_scale))
        dim        = max(dim, 32)

        self.conv_fc1  = nn.Conv2d(D*C, dim, kernel_size = 1, bias = False)
        self.bn_fc1    = nn.BatchNorm2d(dim)
        self.conv_fc2  = nn.Conv2d(dim, 2*D*C, kernel_size = 1, bias = False)
        self.bn_fc2    = nn.BatchNorm2d(2*D*C)

        self.conv3 = nn.Conv2d(D*C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.DC         = D*C
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        d1  = self.relu(self.bn2_d1(self.conv2_d1(out)))
        d2  = self.relu(self.bn2_d2(self.conv2_d2(out)))
        d   = d1 + d2
        #d   = F.avg_pool2d(d, d.size(2))
        d   = self.avg_pool(d)
        d   = self.relu(self.bn_fc1(self.conv_fc1(d)))
        d   = self.bn_fc2(self.conv_fc2(d))
        d   = torch.unsqueeze(d, 1).view(-1, 2, self.DC, 1, 1)
        d   = F.softmax(d, 1)
        d1  = d1 * d[:, 0, :, :, :].squeeze(1)
        d2  = d2 * d[:, 1, :, :, :].squeeze(1)

        out   = d1 + d2
        #out = self.conv2(out)
        #out = self.bn2(out)
        #out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, baseWidth, cardinality, layers, num_classes):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(ResNeXt, self).__init__()
        block = Bottleneck

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AvgPool2d(7)      
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        ret = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        ret["pool1"] = x 
        x = self.layer2(x)
        ret["pool2"] = x
        x = self.layer3(x)
        ret["pool3"] = x
        return ret


def sknet50(baseWidth, cardinality):
    """
    Construct ResNeXt-50.
    """
    model = ResNeXt(baseWidth, cardinality, [3, 4, 6, 3], 1000)
    return model


def sknet101(baseWidth, cardinality):
    """
    Construct ResNeXt-101.
    """
    model = ResNeXt(baseWidth, cardinality, [3, 4, 23, 3], 1000)
    return model


def sknet152(baseWidth, cardinality):
    """
    Construct ResNeXt-152.
    """
    model = ResNeXt(baseWidth, cardinality, [3, 8, 36, 3], 1000)
    return model

class SKEncoder(nn.Module):
    def __init__(self,baseWidth=4,cardinality=32):
        super(SKEncoder,self).__init__()
        self.net = sknet50(baseWidth,cardinality)
    def forward(self,x):
        ret = self.net(x)
        return ret


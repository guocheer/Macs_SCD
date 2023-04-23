import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from core_module import MPFL,MSFL

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Conv2d(512 * block.expansion, 512,kernel_size=3,dilation=2, padding=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight,mode='fan_out')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.embedding(x)
        return out

class MGAM(nn.Module):
    def __init__(self, in_channel= 512):
        super(MGAM, self).__init__()

        self.mpfl = MPFL(in_ch=in_channel // 8, out_ch=in_channel // 8)
        self.msfl = MSFL(in_ch=in_channel //8, out_ch=in_channel // 8)
        self.feature_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channel // 2, out_channels=in_channel, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.emb = nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=1)
        self.conv_1 = nn.Conv2d(in_channels=in_channel,out_channels=1,kernel_size=1)

    def forward(self,x):

        split_num = x.shape[1] // 2
        channel_parts_x = torch.split(x,split_num,dim=1)
        f1_x,f2_x = channel_parts_x[0],channel_parts_x[1]
        f_en_x = self.feature_branch(f1_x)
        f_ls_x = self.mpfl(f2_x)
        f_gc_x = self.msfl(f2_x)
        att_x = torch.sigmoid(self.conv_1(self.emb(torch.cat([f_ls_x,f_gc_x],dim=1))))
        f_att_x = f_en_x * att_x
        return f_att_x

class MGANetSiameseNet(nn.Module):
    def __init__(self):
        super(MGANetSiameseNet, self).__init__()
        self.CNN = ResNet(Bottleneck, [3, 4, 6, 3])
        self.mgam = MGAM(in_channel=512)


    def forward(self, t0,t1):

        feat_t0 = self.CNN(t0)
        feat_t1 = self.CNN(t1)
        f_att_t0_1 = self.mgam(feat_t0)
        f_att_t1_1 = self.mgam(feat_t1)

        f_att_t0_2 = self.mgam(f_att_t0_1)
        f_att_t1_2 = self.mgam(f_att_t1_1)

        f_att_t0_3 = self.mgam(f_att_t0_2)
        f_att_t1_3 = self.mgam(f_att_t1_2)

        dist = F.pairwise_distance(f_att_t0_2,f_att_t1_2,p=2)
        return dist

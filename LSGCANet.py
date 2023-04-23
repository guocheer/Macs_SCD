import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from core_module import MPFL,MSFL

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.embedding1 = nn.Sequential(
            nn.Conv2d(512 * block.expansion, 512, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
        )

        self.embedding2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
        )

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedding1(x)
        x = self.embedding2(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class MGAM(nn.Module):
    def __init__(self, in_channel= 2048):
        super(MGAM, self).__init__()

        self.mpfl = MPFL(in_ch=in_channel // 8, out_ch=in_channel // 8)
        self.msfl = MSFL(in_ch=in_channel // 8, out_ch=in_channel // 8)
        self.feature_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channel // 2, out_channels=in_channel //2, kernel_size=3,stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(in_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel // 2, out_channels=in_channel, kernel_size=3, stride=1, padding=2,dilation=2),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.attention_embed = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=1),
            nn.Conv2d(in_channels=in_channel,out_channels=1,kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self,x):

        split_num = x.shape[1] // 2
        channel_parts_x = torch.split(x,split_num,dim=1)
        f1_x,f2_x = channel_parts_x[0],channel_parts_x[1]
        f_en_x = self.feature_branch(f1_x)
        f_ls_x = self.mpfl(f2_x)
        f_gc_x = self.msfl(f2_x)
        att_x = torch.sigmoid(self.attention_embed(torch.cat([f_ls_x,f_gc_x],dim=1)))
        f_att_x = f_en_x * att_x
        return f_att_x

class LSGCANetSiameseNet(nn.Module):
    def __init__(self):
        super(LSGCANetSiameseNet, self).__init__()
        self.CNN = ResNet(Bottleneck, [3, 4, 6, 3])
        self.mgam = MGAM(in_channel= 512)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, t0,t1):

        feat_t0 = self.CNN(t0)
        feat_t1 = self.CNN(t1)
        f_att_t0_1 =  self.unpool(self.mgam(feat_t0))
        f_att_t1_1 =  self.unpool(self.mgam(feat_t1))
        f_att_t0_2 =  self.unpool(self.mgam(f_att_t0_1))
        f_att_t1_2 =  self.unpool(self.mgam(f_att_t1_1))
        f_att_t0_3 =  self.unpool(self.mgam(f_att_t0_2))
        f_att_t1_3 =  self.unpool(self.mgam(f_att_t1_2))
        f_att_t0_4 =  self.unpool(self.mgam(f_att_t0_3))
        f_att_t1_4 =  self.unpool(self.mgam(f_att_t1_3))
        f_decode_t0_1 = self.unpool(f_att_t0_4)
        f_decode_t1_1 = self.unpool(f_att_t1_4)
        dist = F.pairwise_distance(f_decode_t0_1,f_decode_t1_1,p=2)
        return dist

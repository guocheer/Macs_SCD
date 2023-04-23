import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


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
        #self.embedding = nn.Conv2d(512 * block.expansion, 1024, kernel_size=3,dilation=2, padding=2)
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

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels // 4, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels // 4, output_channels // 4, 3, stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_channels // 4, output_channels, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out

class SAB_AttentionModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SAB_AttentionModule, self).__init__()
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = ResidualBlock(in_channels, out_channels)
        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax3_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )
        self.softmax4_blocks = ResidualBlock(in_channels, out_channels)
        self.softmax5_blocks = ResidualBlock(in_channels, out_channels)
        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels ,kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1 , kernel_size = 1, stride = 1, bias = False),
            nn.Sigmoid()
        )
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, x):
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_interp3 = self.unpool(out_softmax3)
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.unpool(out_softmax4)
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.unpool(out_softmax5)
        out_softmax6 = self.softmax6_blocks(out_interp1)
        return out_softmax6

class MGAM(nn.Module):
    def __init__(self, in_channel= 1024):
        super(MGAM, self).__init__()

        self.sab = SAB_AttentionModule(in_channels= in_channel // 2, out_channels= in_channel // 2)
        self.feature_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channel // 2, out_channels=in_channel//2, kernel_size=3,stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(in_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel // 2, out_channels=in_channel, kernel_size=3, stride=1, padding=2,dilation=2),
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
        att_x = self.sab(f2_x)
        f_att_x = f_en_x * att_x
        return f_att_x

class LSGCANet_SAB_SiameseNet(nn.Module):
    def __init__(self):
        super(LSGCANet_SAB_SiameseNet, self).__init__()
        self.CNN = ResNet(Bottleneck, [3, 4, 6, 3])
        self.mgam = MGAM(in_channel=512)
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

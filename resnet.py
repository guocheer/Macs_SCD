import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import layer.function as fun


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        out = self.embedding(x)
        return out,out,out


class SiameseNet(nn.Module):
    def __init__(self, norm_flag='softmax'):
        super(SiameseNet, self).__init__()
        self.CNN = ResNet(Bottleneck, [3, 4, 6, 3])
        #self.CNN = ResNet(Bottleneck, [3, 4, 23,3])
        if norm_flag == 'l2':
           self.norm = fun.l2normalization(scale=1)
        if norm_flag == 'exp':
            self.norm = nn.Softmax2d()

    def forward(self, t0, t1):

        out_t0_conv5, out_t0_embedding, out_t0_fc8 = self.CNN(t0)
        out_t1_conv5, out_t1_embedding, out_t1_fc8 = self.CNN(t1)
        out_t0_conv5_norm, out_t1_conv5_norm = self.norm(out_t0_conv5), self.norm(out_t1_conv5)
        out_t0_fc8_norm, out_t1_fc8_norm = self.norm(out_t0_fc8), self.norm(out_t1_fc8)
        out_t0_embedding_norm, out_t1_embedding_norm = self.norm(out_t0_embedding), self.norm(out_t1_embedding)

        return [out_t0_conv5_norm, out_t1_conv5_norm], [out_t0_embedding_norm, out_t1_embedding_norm], \
                   [out_t0_fc8_norm, out_t1_fc8_norm]

    def init_parameters(self,pretrain_weights_dir):

        weights = torch.load(pretrain_weights_dir)
        own_state_dict = self.CNN.state_dict()
        pretrain_dicts = {k: v for k, v in weights.items() if k in own_state_dict}
        own_state_dict.update(pretrain_dicts)
        self.CNN.load_state_dict(own_state_dict)
        print('pretrain weights has been loaded sucessfully')

# python3.8
# -- coding: utf-8 --
# -------------------------------
# @Author : &&&&&
# @Email : ￥￥￥￥￥@163.com
# -------------------------------
# @Time : 2023/4/10 15:12
# -------------------------------

def conv_flops(k,c_in,c_out,h,w):
    return k**2 * c_in * c_out * h * w

def pool_flops(k,h,w,c):
    return k ** 2 * h * w * c

def res_b(c_in,c_m,c_out,h,w):
    b1 = conv_flops(1,c_in,c_m,h,w)
    b2 = conv_flops(3,c_m,c_m,h,w)
    b3 = conv_flops(1,c_m,c_out,h,w)
    return b1 + b2 + b3


c1 = conv_flops(7,3,64,224,224)
p1 = pool_flops(3,112,112,64)
c2 = res_b(64,64,256,56,56)
c3 = res_b(256,64,256,56,56)
c4 = res_b(256,256,1024,56,56)
c5 = res_b(1024,512,512,28,28)
c6 = res_b(512,128,512,28,28)
c7 = res_b(512,512,2048,28,28)
c8 = res_b(2048,256,1024,14,14)
c9 = res_b(2048,256,1024,14,14)

print (float(c1 + p1 + c2 + c3 + c4 +c5 +c6 + c7 + c8 +c9) / 1000 ** 3)


import torch
#from torchsummary import summary
from thop import profile

# 构建ResNet50模型
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

# 可视化模型结构
#summary(model, (3, 224, 224))

# 计算模型FLOPs
inputs = torch.randn(1, 3, 512, 512)
flops, params = profile(model, inputs=(inputs, ))

print("ResNet50 FLOPs: {:.2f} GFLOPs".format(flops/1e9))

import torch
#from torchsummary import summary
from thop import profile

class FCN8S(torch.nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8S, self).__init__()

        self.conv1_1 = torch.nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = torch.nn.ReLU(inplace=True)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = torch.nn.ReLU(inplace=True)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = torch.nn.ReLU(inplace=True)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = torch.nn.ReLU(inplace=True)
        self.conv3_2 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = torch.nn.ReLU(inplace=True)
        self.conv3_3 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = torch.nn.ReLU(inplace=True)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1 = torch.nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = torch.nn.ReLU(inplace=True)
        self.conv4_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = torch.nn.ReLU(inplace=True)
        self.conv4_3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = torch.nn.ReLU(inplace=True)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5_1 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = torch.nn.ReLU(inplace=True)
        self.conv5_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = torch.nn.ReLU(inplace=True)
        self.conv5_3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = torch.nn.ReLU(inplace=True)
        self.pool5 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.fc6 = torch.nn.Conv2d(512, 4096, 7)
        self.relu6 = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout2d()

        self.score_fr = torch.nn.Conv2d(4096, num_classes, 1)
        self.score_pool4 = torch.nn.Conv2d(512, num_classes, 1)
        self.score_pool3 = torch.nn.Conv2d(256, num_classes, 1)

        self.upscore2 = torch.nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore_pool4 = torch.nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore8 = torch.nn.ConvTranspose2d(
            num_classes, num_classes, 16, stride=8, bias=False)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.dropout(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h

        h = upscore2 + score_pool4c
        h = self.upscore_pool4(h)
        upscore_pool4 = h

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h

        h = upscore_pool4 + score_pool3c

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

# Instantiate the model
model = FCN8S()

# Input size (3 channels, 512 height, 512 width)
input_size = (3, 512, 512)

# Print the model summary
#summary(model, input_size)

# Calculate the FLOPs
inputs = torch.randn(1, 3, 512, 512)
flops, params = profile(model, inputs=(inputs,))
print("FCN8S Params: {:.2f} M".format(params/1e6))
print("FCN8S FLOPs: {:.2f} GFLOPs".format(flops/1e9))

import torch
#from torchsummary import summary
from thop import profile

# 构建ResNet50模型
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

# 可视化模型结构
#summary(model, (3, 224, 224))

# 计算模型FLOPs
inputs = torch.randn(1, 3, 512, 512)
flops, params = profile(model, inputs=(inputs, ))

print("ResNet50 FLOPs: {:.2f} GFLOPs".format(flops/1e9))

import torch
#from torchsummary import summary
from thop import profile

class FCN8S(torch.nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8S, self).__init__()

        self.conv1_1 = torch.nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = torch.nn.ReLU(inplace=True)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = torch.nn.ReLU(inplace=True)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = torch.nn.ReLU(inplace=True)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = torch.nn.ReLU(inplace=True)
        self.conv3_2 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = torch.nn.ReLU(inplace=True)
        self.conv3_3 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = torch.nn.ReLU(inplace=True)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1 = torch.nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = torch.nn.ReLU(inplace=True)
        self.conv4_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = torch.nn.ReLU(inplace=True)
        self.conv4_3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = torch.nn.ReLU(inplace=True)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5_1 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = torch.nn.ReLU(inplace=True)
        self.conv5_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = torch.nn.ReLU(inplace=True)
        self.conv5_3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = torch.nn.ReLU(inplace=True)
        self.pool5 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        return h

# Instantiate the model
model = FCN8S()

# Input size (3 channels, 512 height, 512 width)
input_size = (3, 512, 512)

# Print the model summary
#summary(model, input_size)

# Calculate the FLOPs
inputs = torch.randn(1, 3, 512, 512)
flops, params = profile(model, inputs=(inputs,))
print("FCN8S Params: {:.2f} M".format(params/1e6))
print("FCN8S FLOPs: {:.2f} GFLOPs".format(flops/1e9))




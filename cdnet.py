import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class convtraction_unit(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(convtraction_unit,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=3,stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class expansion_unit(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(expansion_unit, self).__init__()
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.deconv(x)
        return x

class cdnet_arch(nn.Module):

    def __init__(self):
        super(cdnet_arch, self).__init__()
        self.conv1 = convtraction_unit(in_ch=3,out_ch=64)
        self.conv2 = convtraction_unit(in_ch=64,out_ch=64)
        self.conv3 = convtraction_unit(in_ch=64,out_ch=64)
        self.conv4 = convtraction_unit(in_ch=64, out_ch=64)
        self.deconv4 = expansion_unit(in_ch=64, out_ch=64)
        self.deconv5 = expansion_unit(in_ch=64, out_ch=64)
        self.deconv6 = expansion_unit(in_ch=64, out_ch=64)
        self.deconv7 = expansion_unit(in_ch=64, out_ch=64)
        self.MAC = 0.0

    def cal_mac(self,x):

        return x.data.shape[1]* x.data.shape[2]* x.data.shape[3]

    def forward(self,x):

        x = self.conv1(x)
        self.MAC += self.cal_mac(x)
        x = self.conv2(x)
        self.MAC += self.cal_mac(x)
        x = self.conv3(x)
        self.MAC += self.cal_mac(x)
        x = self.conv4(x)
        self.MAC += self.cal_mac(x)
        x = self.deconv4(x)
        self.MAC += self.cal_mac(x)
        x = self.deconv5(x)
        self.MAC += self.cal_mac(x)
        x = self.deconv6(x)
        self.MAC += self.cal_mac(x)
        x = self.deconv7(x)
        self.MAC += self.cal_mac(x)
        #print("mac is {}".format(2 * self.MAC / (1000 ** 2)))
        return x

class cdnet_SiameseNet(nn.Module):
    def __init__(self):
        super(cdnet_SiameseNet, self).__init__()
        self.cnn = cdnet_arch()
        self.conv = nn.Conv2d(in_channels=128,out_channels=2,kernel_size=7,stride=1,padding=3)
        self.siamese_mac = 0.0

    def forward(self, img1,img2):

        feat_t0 = self.cnn(img1)
        feat_t1 = self.cnn(img2)
        feat_fuse = torch.cat([feat_t0,feat_t1],dim=1)
        score = self.conv(feat_fuse)
        self.siamese_mac = self.cnn.MAC
        #print("mac is {} M".format(self.siamese_mac / (1000 ** 2)))
        return score


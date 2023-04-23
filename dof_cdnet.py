import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

### borrow source code from https://github.com/milesial/Pytorch-UNet ###
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,kernel_size_,stride_):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size= kernel_size_,stride= stride_, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size_,stride_):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch,kernel_size_,stride_)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size_,stride_):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            #nn.MaxPool2d(kernel_size=2,stride=2),
            double_conv(in_ch, out_ch,kernel_size_,stride_)
           )


    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 4, stride=2)
        self.conv = double_conv(in_ch, out_ch, 4, 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class up_inc(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_inc, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 4, 4, stride=2)
        self.conv = double_conv(in_ch //2, out_ch, 4, 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class dof_cdnet_arch(nn.Module):
    def __init__(self):
        super(dof_cdnet_arch, self).__init__()

        self.inc = inconv(3, 64,kernel_size_=3,stride_=1)
        self.down1 = down(64, 128,kernel_size_=4,stride_=2)
        self.down2 = down(128, 256,kernel_size_=4,stride_=2)
        self.down3 = down(256, 512,kernel_size_=4,stride_=2)
        self.down4 = down(512, 512,kernel_size_=4,stride_=2)
        self.down5 = down(512, 512, kernel_size_=4, stride_=2)
        self.down6 = down(512, 512, kernel_size_=4, stride_=2)
        self.down7 = down(512, 512, kernel_size_=4, stride_=2)
        self.up1 = up(1024, 512,bilinear=False)
        self.up2 = up(1024, 512,bilinear=False)
        self.up3 = up(1024, 512,bilinear=False)
        self.up4 = up(1024, 512,bilinear=False)
        self.up5 = up_inc(1024, 256,bilinear=False)
        self.up6 = up_inc(512, 128,bilinear=False)
        self.up7 = up_inc(256, 64,bilinear=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x = self.up1(x8,x7)
        x = self.up2(x,x6)
        x = self.up3(x,x5)
        x = self.up4(x,x4)
        x = self.up5(x,x3)
        x = self.up6(x,x2)
        x = self.up7(x,x1)
        #x = self.up1(x5, x4)
        #x = self.up2(x, x3)
        #x = self.up3(x, x2)
        #x = self.up4(x, x1)
        #x = self.outc(x)
        x_norm = F.normalize(x,p=2,dim=2)
        #return F.sigmoid(x)
        return x_norm

class dof_cdnet_SiameseNet(nn.Module):
    def __init__(self):
        super(dof_cdnet_SiameseNet, self).__init__()
        self.cnn = dof_cdnet_arch()

    def forward(self, img1,img2):

        feat_t0 = self.cnn(img1)
        feat_t1 = self.cnn(img2)
        dist = F.pairwise_distance(feat_t0,feat_t1,p=1)
        return dist



import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class convtraction_unit(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(convtraction_unit,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1,stride=1),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class fc_unit(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size_,padding_):
        super(fc_unit, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size_, padding=padding_,stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
    def forward(self,x):
        x = self.fc(x)
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

class cdnet_fcn_arch(nn.Module):

    def __init__(self):
        super(cdnet_fcn_arch, self).__init__()
        self.conv1 = convtraction_unit(in_ch=3,out_ch=64)
        self.conv2 = convtraction_unit(in_ch=64,out_ch=128)
        self.conv3 = convtraction_unit(in_ch=128,out_ch=256)
        self.conv4 = convtraction_unit(in_ch=256, out_ch=512)
        self.conv5 = convtraction_unit(in_ch=512, out_ch=512)
        self.fc6 = fc_unit(in_ch=512,out_ch=4096,kernel_size_=7,padding_=3)
        self.fc7 = fc_unit(in_ch=4096,out_ch=4096,kernel_size_=1,padding_=0)
        self.fc7_score = nn.Conv2d(4096,2,kernel_size=1,padding=0,stride=1)
        self.fc7_upscore = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1 ,bias=False)
        self.conv4_score = nn.Conv2d(512,2,kernel_size=1,padding=0,stride=1)
        self.conv4_upscore = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1 ,bias=False)
        self.conv3_score = nn.Conv2d(256,2,kernel_size=4,padding=1,stride=1)
        self.conv3_upscore = nn.ConvTranspose2d(2, 2, kernel_size=8, stride=8, padding=0 ,bias=False)

    def forward(self,x):

        x = self.conv1(x)
        x = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        fc6 = self.fc6(c5)
        fc7 = self.fc7(fc6)
        fc7_s = self.fc7_score(fc7)
        fc7_us = self.fc7_upscore(fc7_s)
        c4_s = self.conv4_score(c4)
        c4_fs = fc7_us + c4_s
        c4_us = self.conv4_upscore(c4_fs)
        c3_s = self.conv3_score(c3)
        c3_fs = c3_s + c4_us
        c3_us = self.conv3_upscore(c3_fs)
        return fc7_s


class FCN8s(nn.Module):

    def __init__(self, n_class=2):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4
        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8
        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.MAC = 0.0

    def cal_mac(self,x):

        return x.data.shape[1]* x.data.shape[2]* x.data.shape[3]

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        self.MAC += self.cal_mac(h)
        h = self.relu1_2(self.conv1_2(h))
        self.MAC += self.cal_mac(h)
        h = self.pool1(h)
        self.MAC += self.cal_mac(h)
        h = self.relu2_1(self.conv2_1(h))
        self.MAC += self.cal_mac(h)
        h = self.relu2_2(self.conv2_2(h))
        self.MAC += self.cal_mac(h)
        h = self.pool2(h)
        self.MAC += self.cal_mac(h)
        h = self.relu3_1(self.conv3_1(h))
        self.MAC += self.cal_mac(h)
        h = self.relu3_2(self.conv3_2(h))
        self.MAC += self.cal_mac(h)
        h = self.relu3_3(self.conv3_3(h))
        self.MAC += self.cal_mac(h)
        h = self.pool3(h)
        self.MAC += self.cal_mac(h)
        pool3 = h  # 1/8
        h = self.relu4_1(self.conv4_1(h))
        self.MAC += self.cal_mac(h)
        h = self.relu4_2(self.conv4_2(h))
        self.MAC += self.cal_mac(h)
        h = self.relu4_3(self.conv4_3(h))
        self.MAC += self.cal_mac(h)
        h = self.pool4(h)
        self.MAC += self.cal_mac(h)
        pool4 = h  # 1/16
        h = self.relu5_1(self.conv5_1(h))
        self.MAC += self.cal_mac(h)
        h = self.relu5_2(self.conv5_2(h))
        self.MAC += self.cal_mac(h)
        h = self.relu5_3(self.conv5_3(h))
        self.MAC += self.cal_mac(h)
        h = self.pool5(h)
        self.MAC += self.cal_mac(h)
        h = self.relu6(self.fc6(h))
        self.MAC += self.cal_mac(h)
        h = self.drop6(h)
        h = self.relu7(self.fc7(h))
        self.MAC += self.cal_mac(h)
        h = self.drop7(h)
        h = self.score_fr(h)
        self.MAC += self.cal_mac(h)
        return h

class FCN8_SiameseNet(nn.Module):
    def __init__(self):
        super(FCN8_SiameseNet, self).__init__()
        self.cnn = FCN8s()
        self.conv = nn.Conv2d(in_channels=4,out_channels=2,kernel_size=3,stride=1,padding=1)
        self.siamese_mac = 0.0

    def forward(self, img1,img2):

        feat_t0 = self.cnn(img1)
        feat_t1 = self.cnn(img2)
        res = self.conv(torch.cat([feat_t0,feat_t1],dim=1))
        self.siamese_mac = self.cnn.MAC + res.data.shape[1]* res.data.shape[2]* res.data.shape[3]
        #print("mac is {} M".format(self.siamese_mac / (1000 ** 2)))
        return res



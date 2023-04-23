import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

class deeplab_arch(nn.Module):

    def __init__(self):
        super(deeplab_arch, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode= True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1 ,ceil_mode=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels= 512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1,padding=1,ceil_mode=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, dilation=2 ,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, dilation=2 ,stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, dilation=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        ####### multi-scale contexts #######
        ####### dialtion = 6 ##########
        self.fc6_1 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,dilation=6,padding=6),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 12 ##########
        self.fc6_2 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,dilation=12,padding=12),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 18 ##########
        self.fc6_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=18, padding=18),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 24 ##########
        self.fc6_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=24, padding=24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.embedding_layer = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1)

    def cal_mac(self,x):

        return x.data.shape[1]* x.data.shape[2]* x.data.shape[3]

    def forward(self,x):

        x = self.conv1(x)
        x = self.conv2(x)
        conv3_feature = self.conv3(x)
        conv4_feature = self.conv4(conv3_feature)
        conv5_feature = self.conv5(conv4_feature)
        fc6_1 = self.fc6_1(conv5_feature)
        fc7_1 = self.fc7_1(fc6_1)
        fc6_2 = self.fc6_2(conv5_feature)
        fc7_2 = self.fc7_2(fc6_2)
        fc6_3 = self.fc6_3(conv5_feature)
        fc7_3 = self.fc7_3(fc6_3)
        fc6_4 = self.fc6_4(conv5_feature)
        fc7_4 = self.fc7_4(fc6_4)
        fc_feature = fc7_1 + fc7_2 + fc7_3 + fc7_4
        embedding_feature = self.embedding_layer(fc_feature)
        return embedding_feature

class deeplab_SiameseNet(nn.Module):
    def __init__(self):
        super(deeplab_SiameseNet, self).__init__()
        self.cnn = deeplab_arch()

    def forward(self, img1,img2):

        feat_t0 = self.cnn(img1)
        feat_t1 = self.cnn(img2)
        dist = F.pairwise_distance(feat_t0,feat_t1,p=2)
        return dist
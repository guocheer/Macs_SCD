import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

class MSFL(nn.Module):

    def __init__(self,in_ch=128,out_ch=128):
        super(MSFL, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,dilation=1,stride=1,padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,dilation=2,stride=1,padding=2)

        self.conv2_1 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,dilation=2,stride=1,padding=2)
        self.conv2_2 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,dilation=4,stride=1,padding=4)

        self.conv3_1 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,dilation=4,stride=1,padding=4)
        self.conv3_2 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,dilation=6,stride=1,padding=6)

        self.conv4_1 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,dilation=6,stride=1,padding=6)
        self.conv4_2 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,dilation=8,stride=1,padding=8)

    def forward(self, feat):
        split_num = feat.shape[1] // 4
        channel_parts = torch.split(feat,split_num,dim=1)
        cp1,cp2,cp3,cp4 = channel_parts[0],channel_parts[1],channel_parts[2],channel_parts[3]
        sp1 = self.conv1_2(self.conv1_1(cp1))
        sp2 = self.conv2_2(self.conv2_1(cp2))
        sp3 = self.conv3_2(self.conv3_1(cp3))
        sp4 = self.conv4_2(self.conv4_1(cp4))
        sp = torch.cat([sp1, sp2, sp3, sp4],dim=1)

        return sp

class MPFL(nn.Module):
    def __init__(self,in_ch=128,out_ch=128):
        super(MPFL, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1)
        self.conv1_2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.conv1_3 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.conv1_4 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.conv1_5 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.conv1_6 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.conv1_7 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1)
        self.conv1_8 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.conv1_9 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.conv1_10 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.conv1_11= nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.conv1_12 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.conv1_13 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.conv1_14 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.conv1_15 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.conv1_16 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

        self.conv2_1 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,stride=1,padding=1)
        self.conv2_3 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,stride=1,padding=1)
        self.conv2_4 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,stride=1,padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,stride=1,padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,stride=1,padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,stride=1,padding=1)


    def patch_split(self, input, bin_size):
        """
        b c (bh rh) (bw rw) -> b (bh bw) rh rw c
        """
        B, C, H, W = input.size()
        bin_num_h = bin_size[0]
        bin_num_w = bin_size[1]
        rH = H // bin_num_h
        rW = W // bin_num_w
        out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
        out = out.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, bin_num_h, bin_num_w, rH, rW, C]
        out = out.view(-1,C,rH, rW)  # [B, bin_num_h * bin_num_w, rH, rW, C]
        return out

    def patch_recover(self, input, bin_size):
        """
        b (bh bw) rh rw c -> b c (bh rh) (bw rw)
        """
        B, N, rH, rW, C = input.size()
        bin_num_h = bin_size[0]
        bin_num_w = bin_size[1]
        H = rH * bin_num_h
        W = rW * bin_num_w
        out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, bin_num_h, rH, bin_num_w, rW]
        out = out.view(B, C, H, W)  # [B, C, H, W]
        return out

    def forward(self, feat):
        split_num = feat.shape[1] // 4
        channel_parts = torch.split(feat,split_num,dim=1)
        cp1,cp2,cp3,cp4 = channel_parts[0],channel_parts[1],channel_parts[2],channel_parts[3]
        p1_split = self.patch_split(cp1, (4, 4))
        p2_split = self.patch_split(cp2, (2, 2))
        p3_split = self.patch_split(cp3, (1, 2))
        p4_split = self.patch_split(cp4, (2, 1))
        p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16 = p1_split[0],p1_split[1],p1_split[2],p1_split[3],\
                                                                 p1_split[4],p1_split[5],p1_split[6],p1_split[7],p1_split[8],\
                                                                 p1_split[9],p1_split[10],p1_split[11],p1_split[12],p1_split[13],\
                                                                 p1_split[14],p1_split[15]
        p2_1,p2_2,p2_3,p2_4 = p2_split[0,:],p2_split[1,:],p2_split[2,:],p2_split[3,:]
        p3_1, p3_2 = p3_split[0], p3_split[1]
        p4_1, p4_2 = p4_split[0], p4_split[1]

        p1_c = self.conv1_1(p1.unsqueeze(0))
        p2_c = self.conv1_2(p2.unsqueeze(0))
        p3_c = self.conv1_3(p3.unsqueeze(0))
        p4_c = self.conv1_4(p4.unsqueeze(0))
        p5_c = self.conv1_5(p5.unsqueeze(0))
        p6_c = self.conv1_6(p6.unsqueeze(0))
        p7_c = self.conv1_7(p7.unsqueeze(0))
        p8_c = self.conv1_8(p8.unsqueeze(0))
        p9_c = self.conv1_9(p9.unsqueeze(0))
        p10_c = self.conv1_10(p10.unsqueeze(0))
        p11_c = self.conv1_11(p11.unsqueeze(0))
        p12_c = self.conv1_12(p12.unsqueeze(0))
        p13_c = self.conv1_13(p13.unsqueeze(0))
        p14_c = self.conv1_14(p14.unsqueeze(0))
        p15_c = self.conv1_15(p15.unsqueeze(0))
        p16_c = self.conv1_16(p16.unsqueeze(0))

        p_c_1 = torch.cat([p1_c,p2_c,p3_c,p4_c,p5_c,p6_c,p7_c,p8_c,p9_c,p10_c,p11_c,p12_c,p13_c,p14_c,p15_c,p16_c]).unsqueeze(0).permute(0, 1, 3, 4, 2).contiguous()
        p_r_1 = self.patch_recover(p_c_1,(4,4))

        p2_1_c = self.conv2_1(p2_1.unsqueeze(0))
        p2_2_c = self.conv2_2(p2_2.unsqueeze(0))
        p2_3_c = self.conv2_3(p2_3.unsqueeze(0))
        p2_4_c = self.conv2_4(p2_4.unsqueeze(0))

        p_c_2 = torch.cat([p2_1_c,p2_2_c,p2_3_c,p2_4_c]).unsqueeze(0).permute(0, 1, 3, 4, 2).contiguous()
        p_r_2 = self.patch_recover(p_c_2,(2,2))

        p3_1_c = self.conv3_1(p3_1.unsqueeze(0))
        p3_2_c = self.conv3_2(p3_2.unsqueeze(0))

        p_c_3 = torch.cat([p3_1_c,p3_2_c]).unsqueeze(0).permute(0, 1, 3, 4, 2).contiguous()
        p_r_3 = self.patch_recover(p_c_3,(1,2))

        p4_1_c = self.conv4_1(p4_1.unsqueeze(0))
        p4_2_c = self.conv4_2(p4_2.unsqueeze(0))

        p_c_4 = torch.cat([p4_1_c,p4_2_c]).unsqueeze(0).permute(0, 1, 3, 4, 2).contiguous()
        p_r_4 = self.patch_recover(p_c_4,(2,1))
        p_r = torch.cat([p_r_1,p_r_2,p_r_3,p_r_4],dim=1)
        return p_r
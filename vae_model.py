import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def Normalization(norm_type, out_channels):
    if norm_type==1:
        return nn.InstanceNorm3d(out_channels)
    elif norm_type==2:
        return nn.BatchNorm3d(out_channels,momentum=0.1)


class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),     
            Normalization(norm_type,out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),  
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),
            torch.nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2, activation=True, norm=True):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),
            torch.nn.ReLU(inplace=True)

        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Up(torch.nn.Module):
    def __init__(self, in_ch, out_ch,norm_type=2,kernal_size=(2,2,2),stride=(2,2,2)):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv(in_ch, out_ch, norm_type)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2,kernal_size=(2,2,2),stride=(2,2,2)):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv(in_ch, out_ch, norm_type)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

            


class VAE(torch.nn.Module):

    # [16,32,64,128,256,512]
    def __init__(self, n_class, norm_type=2, n_fmaps=[8,16,32,64,128,256],dim=1024):
        super().__init__()
        self.in_block = Conv(n_class, n_fmaps[0],norm_type=norm_type)
        self.down1 = Down(n_fmaps[0], n_fmaps[1],norm_type=norm_type)
        self.down2 = Down(n_fmaps[1], n_fmaps[2],norm_type=norm_type)
        self.down3 = Down(n_fmaps[2], n_fmaps[3],norm_type=norm_type)
        self.down4 = Down(n_fmaps[3], n_fmaps[4],norm_type=norm_type)
        self.down5 = Down(n_fmaps[4], n_fmaps[5],norm_type=norm_type)
        self.fc_mean = torch.nn.Linear(16384,dim)
        self.fc_std = torch.nn.Linear(16384,dim)
        self.fc2 = torch.nn.Linear(dim,16384)
        self.up1 = Up(n_fmaps[5],n_fmaps[4],norm_type=norm_type)
        self.up2 = Up(n_fmaps[4],n_fmaps[3],norm_type=norm_type)
        self.up3 = Up(n_fmaps[3],n_fmaps[2],norm_type=norm_type)
        self.up4 = Up(n_fmaps[2],n_fmaps[1],norm_type=norm_type)
        self.up5 = Up(n_fmaps[1],n_fmaps[0],norm_type=norm_type)
        self.out_block = torch.nn.Conv3d(n_fmaps[0], n_class, 3, padding=1)
        self.final = nn.Softmax(dim=1)
        self.n_class = n_class
    def forward(self, data_dict,in_key,out_key,if_random=True,scale=1):
        x = data_dict[in_key]
        x = self.in_block(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = x.view(x.size(0),16384)
        x_mean = self.fc_mean(x)
        x_std = nn.ReLU()(self.fc_std(x))
        data_dict['mean'] = x_mean
        data_dict['std'] = x_std
        z = torch.randn(x_mean.size(0),x_mean.size(1)).type(torch.cuda.FloatTensor)
        if if_random:
            x = self.fc2(x_mean+z*x_std*scale)
        else:
            x = self.fc2(x_mean)
        x = x.view(x.size(0),256,4,4,4)
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.out_block(x)
        x = self.final(x)
       
        data_dict[out_key] = x
        return data_dict

        
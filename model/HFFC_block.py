import torch.nn as nn
from FFE import FFE


class HFFC_block(nn.Module):
    def __init__(self,channels):
        super(HFFC_block, self).__init__()
        self.sf_fusion = FFE(inchannels=channels*2)
        self.conv1 = nn.Conv2d(kernel_size=3, in_channels=channels, out_channels=channels,padding=1)
        self.conv2 = nn.Conv2d(kernel_size=1, in_channels=channels, out_channels=2*channels)
        self.conv3 = nn.Conv2d(kernel_size=3, in_channels=channels*2, out_channels=channels*2,padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels*2)
        self.bn3 = nn.BatchNorm2d(channels*2)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2,2)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        x = self.sf_fusion(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu(x)
        x = self.maxpool(x)

        return x

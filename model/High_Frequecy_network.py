import torch
import torch.nn as nn
from HFFC_block import HFFC_block


class High_Frequency_network(nn.Module):
    def __init__(self):
        super(High_Frequency_network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32
                               , kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.Leakyrelu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.High_block1 = HFFC_block(32)
        self.High_block2 = HFFC_block(64)
        self.High_block3 = HFFC_block(128)
        self.High_block4 = HFFC_block(256)
        self.Adaptive_maxpool = nn.AdaptiveMaxPool2d(1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.Leakyrelu(x)
        x = self.High_block1(x)
        x = self.High_block2(x)
        x = self.High_block3(x)
        x = self.High_block4(x)
        x = self.Adaptive_maxpool(x)
        x = torch.flatten(x, 1)
        return x
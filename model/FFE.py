
import torch
import torch.nn as nn
import torch.nn.functional as F


class Frequency_dynamic_depose(nn.Module):
    def __init__(self, features, r=16, M=2) -> None:
        super().__init__()

        self.features = features
        self.relu = nn.ReLU(inplace=True)
        self.fc_low = nn.Conv2d(features, features // r, 1, 1, 0)
        self.fc_high = nn.Conv2d(features, features // r, 1, 1, 0)
        self.bn_low_1 = nn.BatchNorm2d(features // r)
        self.bn_low_2 = nn.BatchNorm2d(features)
        self.bn_high_1 = nn.BatchNorm2d(features // r)
        self.bn_high_2 = nn.BatchNorm2d(features)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(features // r, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, low, high):
        input_low = low
        input_high = high
        low = self.gap(low)
        low = self.fc_low(low)
        low = self.bn_low_1(low)
        low = self.relu(low)
        low = self.fcs[0](low)
        low = self.bn_low_2(low)
        low_att = self.softmax(low)

        high = self.gap(high)
        high = self.fc_high(high)
        high = self.bn_high_1(high)
        high = self.relu(high)
        high = self.fcs[1](high)
        high = self.bn_high_2(high)
        high_att = self.softmax(high)

        fea_high = input_high * high_att + input_high
        fea_low = input_low * low_att + low + input_low

        return fea_low, fea_high


class FFE(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(FFE, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))

        self.modulate = Frequency_dynamic_depose(inchannels)
        self.fusion = Global_fusion(inchannels)

    def forward(self, x):
        identity_input = x  # 3,32,64,64
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size)
        x = x.reshape(n, self.group, c // self.group,
                      self.kernel_size ** 2, h * w)

        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)

        low_filter = self.act(low_filter)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        se_low, se_high = self.modulate(low_part, out_high)
        out = self.fusion(se_low, se_high)

        return out


class Global_fusion(nn.Module):
    
    def __init__(self, channels=64, r=16):
        super(Global_fusion, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )


        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        wei = self.sigmoid(xl)
        xo = x * wei + residual * (1 - wei) +x+residual
        return xo

if __name__ == '__main__':
    model = FFE(32)
    x = torch.randn((32,32,224,224))
    y = model(x)
    print(y.shape)

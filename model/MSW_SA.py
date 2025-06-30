from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import wave_conv


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, wt_levels=4, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels
        self.bias =bias
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = wave_conv.create_wavelet_filter(wt_type, in_channels, in_channels,
                                                                        torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wave_conv.wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(wave_conv.inverse_wavelet_transform, filters=self.iwt_filter)



        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

        self.base_convs = nn.ModuleList([nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, padding='same', stride=1, dilation=1,
                                       groups=self.in_channels,
                                       bias=self.bias) for _ in range(self.wt_levels)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.in_channels) for _ in range(self.wt_levels)])
        self.base_scales = nn.ModuleList(
            [_ScaleModule([1, self.in_channels, 1, 1]) for _ in range(self.wt_levels)]
        )  #

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):

            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(
                curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3],
                                        shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])


        out = 0
        for i in range(self.wt_levels,0,-1):
            next_x_ll = 0
            base_conv = self.base_convs[i - 1]
            base_scale = self.base_scales[i - 1]
            bn = self.bns[i - 1]  #

            count = i


            for j in range(count - 1, -1, -1):


                curr_x_ll = x_ll_in_levels[j]
                curr_x_h = x_h_in_levels[j]
                curr_shape = shapes_in_levels[j]

                curr_x_ll = curr_x_ll + next_x_ll

                curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
                next_x_ll = self.iwt_function(curr_x)

                next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

            x_tag = next_x_ll


            y = base_scale(base_conv(x))
            y = bn(y + x_tag)






            if self.do_stride is not None:
              y = self.do_stride(x)
            out = out+y

        return out
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class SpatialAttention(nn.Module):
    def __init__(self,kernel_size,wt_levels):
        super().__init__()
        self.sa = nn.Sequential()




        self.sa.add_module('conv_%d', WTConv2d(in_channels=2, out_channels=2,kernel_size=kernel_size,bias=True,wt_levels=wt_levels,wt_type='db1'))
        self.sa.add_module('last_conv', nn.Conv2d(2, 1, kernel_size=1))


    def forward(self, x):
        max_result,_ = torch.max(x,dim=1,keepdim=True)
        avg_result= torch.mean(x,dim=1,keepdim=True)
        res = torch.cat([max_result,avg_result],dim=1)
        res = self.sa(res)
        res = res.expand_as(x)
        return res


class MSW_SA(nn.Module):

    def __init__(self,kernel_size=3,wt_levels=2,):
        super().__init__()

        self.sa = SpatialAttention(kernel_size=kernel_size,wt_levels=wt_levels )
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        sa_out = self.sa(x)
        weight = self.sigmoid(sa_out)
        out = (1 + weight) * x
        return out

import torch
import torch.nn as nn

def get_candidates(peaks,
                   channels,
                   stride=1):


    if stride == 1:

        candidates = nn.ModuleList([
            ZeroOperation(stride=stride),
            nn.Identity()])

        for peak in peaks:
            padding = (peak - 1) // 2
            candidates.append(MaxPoolOperation(channels, peak, stride=stride, padding=padding))
            candidates.append(AvgPoolOperation(channels, peak, stride=stride, padding=padding))
            candidates.append(SeparableOperation(channels, channels, peak, stride=stride, padding=padding, repeat=True))
            candidates.append(SeparableOperation(channels, channels, peak, stride=stride, padding=padding, dilation=2))

    elif stride == 4:
        candidates = nn.ModuleList([
            ZeroOperation(stride=stride),
            FactorizedReduction(channels, channels)])

        for peak in peaks:
            padding = (peak - 1) // 2
            candidates.append(MaxPoolOperation(channels, peak, stride=stride, padding=padding))
            candidates.append(AvgPoolOperation(channels, peak, stride=stride, padding=padding))
            candidates.append(SeparableOperation(channels, channels, peak, stride=stride, padding=padding, repeat=True))
            candidates.append(SeparableOperation(channels, channels, peak, stride=stride, padding=padding, dilation=2))

    return candidates

class ReLUConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        super(ReLUConvBN, self).__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm1d(out_channels, affine=False)

    def forward(self, input):
        output = self.relu(input)
        output = self.conv(output)
        output = self.bn(output)

        return output

class ReLUSepConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super(ReLUSepConvBN, self).__init__()

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels,
                      in_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=in_channels,
                      bias=bias),
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size=1,
                      stride=1,
                      bias=bias),
            nn.BatchNorm1d(out_channels, affine=False)
        )

    def forward(self, input):
        return self.layers(input)

class SeparableOperation(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 repeat=False):
        super(SeparableOperation, self).__init__()

        if repeat:
            self.layers = nn.Sequential(
                ReLUSepConvBN(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias),
                ReLUSepConvBN(in_channels,
                              out_channels,
                              kernel_size,
                              stride=1,
                              padding=padding,
                              bias=bias)
                )
        else:
            self.layers = ReLUSepConvBN(in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        bias=bias)

    def forward(self, input):
        output = self.layers(input)
        return output

class DilatedSeparableOperation(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super(SeparableOperation, self).__init__()

        self.layers = ReLUSepConvBN(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation)

    def forward(self, input):
        output = self.layers(input)
        return output


class ZeroOperation(nn.Module):
    def __init__(self,
                 stride=1):
        super(ZeroOperation, self).__init__()
        self.stride = stride

    def forward(self, input):
        if self.stride == 1:
            return input.mul(0.)
        else:
            return input[:, :, ::self.stride].mul(0.)

class MaxPoolOperation(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 ceil_mode=False):
        super(MaxPoolOperation, self).__init__()

        self.layers = nn.Sequential(
            nn.MaxPool1d(kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode),
            nn.BatchNorm1d(channels, affine=False)
        )

    def forward(self, input):
        return self.layers(input)

class AvgPoolOperation(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride=None,
                 padding=0,
                 ceil_mode=False):
        super(AvgPoolOperation, self).__init__()

        self.layers = nn.Sequential(
            nn.AvgPool1d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode),
            nn.BatchNorm1d(channels, affine=False)
        )

    def forward(self, input):
        return self.layers(input)



class FactorizedReduction(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=4):
        super(FactorizedReduction, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels,
                               out_channels // 2,
                               kernel_size=1,
                               stride=stride,
                               bias=False)
        self.conv2 = nn.Conv1d(in_channels,
                               out_channels // 2,
                               kernel_size=1,
                               stride=stride,
                               bias=False)
        self.bn = nn.BatchNorm1d(out_channels, affine=False)

    def forward(self, input):

        output = self.relu(input)
        output = torch.cat([self.conv1(output), self.conv2(output[:, :, 1:])], dim=1)
        output = self.bn(output)

        return output
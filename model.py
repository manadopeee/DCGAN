import torch
from layer import *

# DCGAN
# https://arxiv.org/pdf/1511.06434.pdf

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=128):
        super(Generator, self).__init__()

        self.deconv_1 = DECOV2d(in_channels=in_channels, out_channels=nker * 8, kernel_size=5, stride=1, padding=0, bias=False, norm=True, relu=0.0)
        self.deconv_2 = DECOV2d(in_channels=nker * 8, out_channels=nker * 4, kernel_size=5, stride=2, padding=1, bias=False, norm=True, relu=0.0)
        self.deconv_3 = DECOV2d(in_channels=nker * 4, out_channels=nker * 2, kernel_size=5, stride=2, padding=1, bias=False, norm=True, relu=0.0)
        self.deconv_4 = DECOV2d(in_channels=nker * 2, out_channels=nker * 1, kernel_size=5, stride=2, padding=1, bias=False, norm=True, relu=0.0)
        self.deconv_5 = DECOV2d(in_channels=nker * 1, out_channels=out_channels, kernel_size=5, stride=2, padding=1, bias=False, norm=False, relu=None)

    def forward(self, x):
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        x = self.deconv_5(x)

        x = torch.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=128):
        super(Generator, self).__init__()

        self.conv_1 = COV2d(in_channels=in_channels, out_channels=nker * 1, kernel_size=5, stride=2, padding=1, bias=False, norm=True, relu=0.2)
        self.conv_2 = COV2d(in_channels=nker * 1, out_channels=nker * 2, kernel_size=5, stride=2, padding=1, bias=False, norm=True, relu=0.2)
        self.conv_3 = COV2d(in_channels=nker * 2, out_channels=nker * 4, kernel_size=5, stride=2, padding=1, bias=False, norm=True, relu=0.2)
        self.donv_4 = COV2d(in_channels=nker * 4, out_channels=nker * 8, kernel_size=5, stride=2, padding=1, bias=False, norm=True, relu=0.2)
        self.conv_5 = COV2d(in_channels=nker * 8, out_channels=out_channels, kernel_size=5, stride=1, padding=0, bias=False, norm=False, relu=None)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)

        x = torch.sigmoid(x)

        return x
import torch.nn as nn

class DECOV2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=1, bias=False, norm=True, relu=0.0):
        layers = []
        layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if norm:
            layers += [nn.BatchNorm2d(num_features=out_channels)]
        if relu == 0.0:
            layers += [nn.ReLU()]

        self.decov = nn.Sequential(*layers)

    def forward(self, x):
        return self.decov(x)


class COV2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=1, bias=False, norm=True, relu=0.2):
        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if norm:
            layers += [nn.BatchNorm2d(num_features=out_channels)]
        if relu:
            layers += [nn.LeakyReLU(relu)]

        self.cov = nn.Sequential(*layers)

    def forward(self, x):
        return self.cov(x)
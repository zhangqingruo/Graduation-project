import torch
import torch.nn as nn
import torch.nn.functional as F


# Generator
class Generator(nn.Module):
    def __init__(self, channels=1):
        super(Generator, self).__init__()

        self.channels = channels
        self.block1 = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.resnet_blocks = nn.Sequential(*[ResNetBlock(256) for _ in range(30)])

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resnet_blocks(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        return x


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = out + identity

        return out


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels=1):
        super(Discriminator, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            self.conv_bn_lrelu(64, 128),
            self.conv_bn_lrelu(128, 256),
            self.conv_bn_lrelu(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def conv_bn_lrelu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.l1(x)

        return x

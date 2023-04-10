import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_BN=True):
        super(ConvBlock, self).__init__()
        if use_BN:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)


class AuxiliaryOutput(nn.Module):
    def __init__(self, in_channels):
        super(AuxiliaryOutput, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=14, stride=1, padding=0, bias=True)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return x


# class UNet(nn.Module):
#     def __init__(self, use_BN=True):
#         super(UNet, self).__init__()
#         self.conv1 = ConvBlock(1, 64, use_BN=use_BN)
#         self.conv2 = ConvBlock(64, 128, use_BN=use_BN)
#         self.conv3 = ConvBlock(128, 256, use_BN=use_BN)
#         self.conv4 = ConvBlock(256, 512, use_BN=use_BN)
#         self.conv5 = ConvBlock(512, 1024, use_BN=use_BN)
#
#         # self.aux = AuxiliaryOutput(1024)
#
#         self.upsample54 = UpSample(1024, 512)
#         self.conv4m = ConvBlock(1024, 512, use_BN=use_BN)
#
#         self.upsample43 = UpSample(512, 256)
#         self.conv3m = ConvBlock(512, 256, use_BN=use_BN)
#
#         self.upsample32 = UpSample(256, 128)
#         self.conv2m = ConvBlock(256, 128, use_BN=use_BN)
#
#         self.upsample21 = UpSample(128, 64)
#         self.conv1m = ConvBlock(128, 64, use_BN=use_BN)
#
#         self.conv0 = nn.Conv2d(64, 1, kernel_size=1)
#
#     def forward(self, x):
#         conv1_out = self.conv1(x)
#         conv2_out = self.conv2(nn.functional.max_pool2d(conv1_out, 2))
#         conv3_out = self.conv3(nn.functional.max_pool2d(conv2_out, 2))
#         conv4_out = self.conv4(nn.functional.max_pool2d(conv3_out, 2))
#         conv5_out = self.conv5(nn.functional.max_pool2d(conv4_out, 2))
#
#         # aux_out = self.aux(conv5_out)
#         conv5m_out = torch.cat([self.upsample54(conv5_out), conv4_out], dim=1)
#         conv4m_out = self.conv4m(conv5m_out)
#
#         conv4m_out = torch.cat([self.upsample43(conv4m_out), conv3_out], dim=1)
#         conv3m_out = self.conv3m(conv4m_out)
#
#         conv3m_out = torch.cat([self.upsample32(conv3m_out), conv2_out], dim=1)
#         conv2m_out = self.conv2m(conv3m_out)
#
#         conv2m_out = torch.cat([self.upsample21(conv2m_out), conv1_out], dim=1)
#         conv1m_out = self.conv1m(conv2m_out)
#
#         conv0_out = self.conv0(conv1m_out)
#         return conv0_out


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        # self.aux = AuxiliaryOutput(1024)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

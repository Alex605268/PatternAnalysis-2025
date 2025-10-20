# Contains the source code of the model.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        #Very basic ConvBlock, based on lecture example
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p  # return features before pooling for skip connection

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)  # in_channels includes skip connection

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # concatenate along channel dimension
        x = self.conv(x)
        return x



class UNet(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=3, base_filters=64):
        super(UNet, self).__init__()
        # Encoder
        self.down1 = DownBlock(in_channels, base_filters)
        self.down2 = DownBlock(base_filters, base_filters*2)
        self.down3 = DownBlock(base_filters*2, base_filters*4)
        self.down4 = DownBlock(base_filters*4, base_filters*8)

        # Bottleneck
        self.bottleneck = ConvBlock(base_filters*8, base_filters*16)

        # Decoder
        self.up4 = UpBlock(base_filters*16, base_filters*8)
        self.up3 = UpBlock(base_filters*8, base_filters*4)
        self.up2 = UpBlock(base_filters*4, base_filters*2)
        self.up1 = UpBlock(base_filters*2, base_filters)

        # Final conv
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        pass

    def forward(self, x):
        # Encoder
        s1, p1 = self.down1(x)
        s2, p2 = self.down2(p1)
        s3, p3 = self.down3(p2)
        s4, p4 = self.down4(p3)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        d4 = self.up4(b, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        out = self.final_conv(d1)
        return out


def build_unet(in_channels=1, out_channels=3):
    return UNet(in_channels, out_channels)


#Forward pass test
if __name__ == "__main__":
    x = torch.randn(1, 1, 256, 256)  # batch_size=1, channels=1, H=W=256
    model = build_unet()
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)  # should be [1, out_channels, 256, 256]

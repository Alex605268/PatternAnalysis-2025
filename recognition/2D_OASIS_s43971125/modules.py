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

'''
Basic UNet - this code was adapted from the Lectures
'''

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, base_filters=64):
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


#Below are all the requirements for the improved UNet


class PreActResBlock(nn.Module):
    #dropout_prob is optional tuning metric, used to regularise the block (prevents overfitting) 
    def __init__(self, in_ch, out_ch, dropout_prob=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dropout_prob = dropout_prob

        #First Normalisation + convolution
        self.norm1 = nn.InstanceNorm2d(in_ch, affine=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

        #Second normalisation + convolution
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)

        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        
        #check that input can be added to output
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.norm2(out)))
        # residual add
        return out + self.skip(x)

class DownResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_prob=0.0):
        super().__init__()
        self.res = PreActResBlock(in_ch, out_ch, dropout_prob=dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        features = self.res(x)
        pooled = self.pool(features)
        return features, pooled

class UpResBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout_prob=0.0):
        """
        in_ch = channels of decoder input (from previous layer)
        skip_ch = channels from encoder skip connection
        out_ch = desired output channels after block
        First upsample (in_ch -> out_ch), then concatenate with skip (skip_ch),
        then use a PreActResBlock with in_ch = out_ch + skip_ch, out_ch = out_ch
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = PreActResBlock(out_ch + skip_ch, out_ch, dropout_prob=dropout_prob)

    def forward(self, x, skip):
        x = self.up(x)
        # If shapes differ by one pixel due to odd sizes, center-crop or pad (common safe approach)
        if x.shape[-2:] != skip.shape[-2:]:
            # simple center crop/pad to match skip spatial dims
            target_h, target_w = skip.shape[-2], skip.shape[-1]
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

"""
Improved UNet:
- PreAct residual blocks
- InstanceNorm
- Dropout2d
- Deep supervision
Parameters:
in_channels: input channels (e.g. 1)
out_channels: number of segmentation classes (4 for 2D_OASIS)
base_filters: number of filters at first level (commonly 32 or 64)
dropout_prob: probability for spatial dropout inside blocks
deep_supervision: whether to include deep supervision
"""
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, base_filters=64,
                 dropout_prob=0.1, deep_supervision=True):
        super().__init__()
        f = base_filters
        self.deep_supervision = deep_supervision

        # Encoder
        self.enc1 = DownResBlock(in_channels, f, dropout_prob=dropout_prob)
        self.enc2 = DownResBlock(f, f*2, dropout_prob=dropout_prob)
        self.enc3 = DownResBlock(f*2, f*4, dropout_prob=dropout_prob)
        self.enc4 = DownResBlock(f*4, f*8, dropout_prob=dropout_prob)

        # Bottleneck
        self.bottleneck = PreActResBlock(f*8, f*16, dropout_prob=dropout_prob)

        # Decoder (note channel bookkeeping)
        self.up4 = UpResBlock(f*16, skip_ch=f*8, out_ch=f*8, dropout_prob=dropout_prob)
        self.up3 = UpResBlock(f*8, skip_ch=f*4, out_ch=f*4, dropout_prob=dropout_prob)
        self.up2 = UpResBlock(f*4, skip_ch=f*2, out_ch=f*2, dropout_prob=dropout_prob)
        self.up1 = UpResBlock(f*2, skip_ch=f,   out_ch=f,   dropout_prob=dropout_prob)

        # Final 1x1 conv to logits
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

        # Auxiliary heads for deep supervision - map intermediate decoder features to logits
        if self.deep_supervision:
            self.aux4 = nn.Conv2d(f*8, out_channels, kernel_size=1)  # from d4
            self.aux3 = nn.Conv2d(f*4, out_channels, kernel_size=1)  # from d3
            self.aux2 = nn.Conv2d(f*2, out_channels, kernel_size=1)  # from d2
            # We do not need aux for d1 since final_conv handles it

    def forward(self, x):
        # Encoder Step
        # each call of self.enc(1-4) is an instance of DownResBlock
        #s1-4 are saved so they can be used for skip connections
        #p1-4 are the progressively downsampled features of the input
        s1, p1 = self.enc1(x)   # s1: f
        s2, p2 = self.enc2(p1)  # s2: f*2
        s3, p3 = self.enc3(p2)  # s3: f*4
        s4, p4 = self.enc4(p3)  # s4: f*8

        # Bottleneck
        # This is the deepest layer of the network (i.e we encode down to here, do feature extraction 
        # at the lowest level (most abstracted features), then decode back up to higher level.
        b = self.bottleneck(p4)  # f*16

        # Decoder
        #Each call of up(1-4) is an instance of upResBlock
        #it takes a feature map and a skip connection, upsamples the feature map, concatenates
        #it with the skip connection, then merges the features.
        # each call utilises the previous feature map that has been upsampled
        d4 = self.up4(b, s4)  # f*8
        d3 = self.up3(d4, s3) # f*4
        d2 = self.up2(d3, s2) # f*2
        d1 = self.up1(d2, s1) # f

        # Final convolution
        # We take our final feature map from the decode, and map it to the number of output classes
        # we can then feed this result into DiceLoss
        out_final = self.final_conv(d1)  # [B, out_channels, H, W]

        if not self.deep_supervision:
            return out_final

        # Deep supervision
        # create extra predictions at lower decoder levels (i.e not d1)
        # These are essentially "early" predictions, at lower resolutions
        aux4 = self.aux4(d4)
        aux3 = self.aux3(d3)
        aux2 = self.aux2(d2)

        # Upsample Aux
        # because each decoder layer has a different spatial size, have to upsample all of them so they match
        # out_final's size
        target_size = out_final.shape[-2:]
        aux4_up = F.interpolate(aux4, size=target_size, mode='bilinear', align_corners=False)
        aux3_up = F.interpolate(aux3, size=target_size, mode='bilinear', align_corners=False)
        aux2_up = F.interpolate(aux2, size=target_size, mode='bilinear', align_corners=False)

        # Sum elementwise to combine them. 
        # By doing this, gives stronger supervision to early layers
        # also means we can swap out standard UNet and Improved UNet in train.py with no changes
        # since they both return a single output this way
        combined = out_final + aux2_up + aux3_up + aux4_up
        return combined



def build_unet(in_channels=1, out_channels=4):
    return UNet(in_channels, out_channels)

def build_improved_unet(in_channels=1, out_channels=4, base_filters=32, dropout_prob=0.1, deep_supervision=True):
    return ImprovedUNet(in_channels=in_channels, out_channels=out_channels,
                        base_filters=base_filters, dropout_prob=dropout_prob,
                        deep_supervision=deep_supervision)

#Forward pass test
if __name__ == "__main__":
    x = torch.randn(1, 1, 256, 256)  # batch_size=1, channels=1, H=W=256
    model = build_unet()
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)  # should be [1, out_channels, 256, 256]

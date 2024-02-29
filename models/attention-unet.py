
import torch
import torch.nn as nn
from torchinfo import summary

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class attention_gate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = attention_gate(in_c, out_c)
        self.c1 = conv_block(in_c[0]+out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x

class attention_unet(nn.Module):
    def __init__(self, in_channels, n_classes, level_channels, bottleneck_channel):
        super().__init__()

        self.e1 = encoder_block(in_channels, level_channels[0])
        self.e2 = encoder_block(level_channels[0], level_channels[1])
        self.e3 = encoder_block(level_channels[1], level_channels[2])

        self.b1 = conv_block(level_channels[2], bottleneck_channel)

        self.d1 = decoder_block([bottleneck_channel, level_channels[2]], level_channels[2])
        self.d2 = decoder_block([level_channels[2], level_channels[1]], level_channels[1])
        self.d3 = decoder_block([level_channels[1], level_channels[0]], level_channels[0])

        self.output = nn.Conv2d(level_channels[0], n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b1 = self.b1(p3)

        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        output = self.output(d3)
        return output


if __name__ == "__main__":
    
    inputs = torch.randn((2, 3, 256, 256))
    in_channels = 3
    n_classes = 6
    level_channels = [64, 128, 256]
    bottleneck_channel = 512

    model = attention_unet(in_channels = in_channels , n_classes = n_classes, \
        level_channels = level_channels, bottleneck_channel = bottleneck_channel)
    
    summary(model, input_size=inputs.shape)

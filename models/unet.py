import torch
import torch.nn as nn


from torchinfo import summary

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""

class conv2d_block(nn.Module):
    
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels= in_c, out_channels=out_c//2, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(out_c //2)

        self.conv2 = nn.Conv2d(out_c//2, out_c, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder2d_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv2d_block(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder2d_block(nn.Module):
    def __init__(self, in_c, out_c, upsample, upsample_mode ):
        super().__init__()

        # if upsample, use the normal convolutions to reduce the number of channels
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True) # Upsample is used to increase the resolution of the image and the number of channels is not modified
            self.conv = conv2d_block(in_c+out_c, out_c)
        else:
            self.up = nn.ConvTranspose2d(in_c, in_c, kernel_size=2, stride=2, padding=0)
            self.conv = conv2d_block(in_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        # print("up_shape = ", x.shape,"skip_shape = ", skip.shape)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class Unet(nn.Module):
    def __init__(self, in_channels, n_classes, upsample,upsample_mode, level_channels, bottleneck_channel):
        super().__init__()

        """ Encoder """
        self.e1 = encoder2d_block(in_channels, level_channels[0])
        self.e2 = encoder2d_block(level_channels[0], level_channels[1])
        self.e3 = encoder2d_block(level_channels[1], level_channels[2])
        self.e4 = encoder2d_block(level_channels[2], level_channels[3])

        """ Bottleneck """
        self.b = conv2d_block(level_channels[3], bottleneck_channel)

        """ Decoder """
        self.d1 = decoder2d_block(bottleneck_channel, level_channels[3], upsample, upsample_mode )
        self.d2 = decoder2d_block(level_channels[3], level_channels[2], upsample, upsample_mode )
        self.d3 = decoder2d_block(level_channels[2], level_channels[1], upsample, upsample_mode )
        self.d4 = decoder2d_block(level_channels[1], level_channels[0], upsample, upsample_mode )

        """ Classifier """
        self.outputs = nn.Conv2d(level_channels[0], n_classes, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs

if __name__ == "__main__":
    
    inputs = torch.randn((2, 3, 256, 256))
    in_channels = 3
    n_classes = 6
    level_channels = [64, 128, 256, 512]
    bottleneck_channel = 1024
    upsample = True
    upsample_mode = 'bilinear'
    

    print("Test when upsample = True\n")
    model = Unet(in_channels = in_channels , n_classes = n_classes, upsample=upsample, upsample_mode = upsample_mode, level_channels = level_channels, bottleneck_channel = bottleneck_channel)
    summary(model, input_size=inputs.shape)
    
    print("Test when upsample = False\n")
    model = Unet(in_channels = in_channels , n_classes = n_classes, upsample=upsample, upsample_mode = upsample_mode, level_channels = level_channels, bottleneck_channel = bottleneck_channel)
    summary(model, input_size=inputs.shape)
    

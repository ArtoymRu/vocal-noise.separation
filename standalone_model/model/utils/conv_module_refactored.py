import torch
import torch.nn as nn
import torch.nn.functional as F

class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)

class DepthwiseConv1d(nn.Module):
    """ Depthwise 1D convolution. """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels)

    def forward(self, x):
        return self.conv(x)

class ConvolutionBlock(nn.Module):
    """ A block that combines convolutions with normalization and activation layers. """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=nn.ReLU(), norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            norm_layer(out_channels),
            activation
        )

    def forward(self, x):
        return self.seq(x)

# Example usage:
if __name__ == "__main__":
    # Create a sample tensor of shape (batch_size, channels, length)
    x = torch.randn(10, 32, 100)
    # Apply the ConvolutionBlock
    conv_block = ConvolutionBlock(32, 64, kernel_size=3, padding=1)
    output = conv_block(x)
    print(output.shape)

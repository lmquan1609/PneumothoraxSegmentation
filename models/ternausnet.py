from torch import nn
import torch.nn.functional as F
from torchvision import models

import torch
import torchvision

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels=input_channel, out_channels=out_channels, 
                    kernel_size=3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.activation = nn.ReLU(inplace=True)
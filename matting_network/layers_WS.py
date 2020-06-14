###############################################################################
# Original source code for the FBA Matting model used in the refinement stage 
# can be found at https://github.com/MarcoForte/FBA_Matting
#
# Pre-trained model weights can also be found at the url above. Note that these
# are covered by the Deep Image Matting Dataset License Agreement for which
# the reader should refer to https://sites.google.com/view/deepimagematting
#
# This code leverages the FBA matting model trained with GroupNorm and Weight
# Standardisation specifically (filenames are preserved as per the source)
###############################################################################

# external libraries
import torch
import torch.nn as nn
from torch.nn import functional as F


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def BatchNorm2d(num_features):
    return nn.GroupNorm(num_channels=num_features, num_groups=32)

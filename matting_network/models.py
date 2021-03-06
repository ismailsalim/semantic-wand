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

# local application libraries
from matting_network.resnet_GN_WS import Bottleneck, ResNet, ResNetDilated
import matting_network.layers_WS as L

# external libraries
import torch
import torch.nn as nn


def build_model(weights):
    model = MattingModule()
    model.cuda()

    sd = torch.load(weights)
    model.load_state_dict(sd, strict=True)

    return model


class MattingModule(nn.Module):
    def __init__(self):
        super(MattingModule, self).__init__()
        self.encoder = build_encoder()
        self.decoder = FBA_Decoder()

    def forward(self, image, two_chan_trimap, image_norm, trimap_transformed):
        resnet_input = torch.cat((image_norm, trimap_transformed, two_chan_trimap), 1)
        conv_out, indices = self.encoder(resnet_input, return_feature_maps=True)
        return self.decoder(conv_out, image, indices, two_chan_trimap)


def build_encoder():
    orig_resnet = ResNet(Bottleneck, [3, 4, 6, 3])
    net_encoder = ResNetDilated(orig_resnet, dilate_scale=8)

    num_channels = 3 + 6 + 2

    net_encoder_sd = net_encoder.state_dict()
    conv1_weights = net_encoder_sd['conv1.weight']

    c_out, c_in, h, w = conv1_weights.size()
    conv1_mod = torch.zeros(c_out, num_channels, h, w)
    conv1_mod[:, :3, :, :] = conv1_weights

    conv1 = net_encoder.conv1
    conv1.in_channels = num_channels
    conv1.weight = torch.nn.Parameter(conv1_mod)

    net_encoder.conv1 = conv1

    net_encoder_sd['conv1.weight'] = conv1_mod

    net_encoder.load_state_dict(net_encoder_sd)
    return net_encoder


class FBA_Decoder(nn.Module):
    def __init__(self):
        super(FBA_Decoder, self).__init__()
        pool_scales = (1, 2, 3, 6) 
        self.ppm = []

        for scale in pool_scales: # pyramid pooling
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                L.Conv2d(2048, 256, kernel_size=1, bias=True),
                nn.GroupNorm(32, 256),
                nn.LeakyReLU()
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_up1 = nn.Sequential(
            L.Conv2d(2048 + len(pool_scales) * 256, 256,
                     kernel_size=3, padding=1, bias=True),

            nn.GroupNorm(32, 256),
            nn.LeakyReLU(),
            L.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU()
        )

        self.conv_up2 = nn.Sequential(
            L.Conv2d(256 + 256, 256,
                     kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU()
        )
        
        d_up3 = 64
        self.conv_up3 = nn.Sequential(
            L.Conv2d(256 + d_up3, 64,
                     kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU()
        )

        self.unpool = nn.MaxUnpool2d(2, stride=2)

        self.conv_up4 = nn.Sequential(
            nn.Conv2d(64 + 3 + 3 + 2, 32,
                      kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16,
                      kernel_size=3, padding=1, bias=True),

            nn.LeakyReLU(),
            nn.Conv2d(16, 7, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, conv_out, img, indices, two_chan_trimap):
        conv5 = conv_out[-1] 

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm: # pyramid pooling
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_up1(ppm_out)

        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x, conv_out[-4]), 1)

        x = self.conv_up2(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x, conv_out[-5]), 1)
        x = self.conv_up3(x)

        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, conv_out[-6][:, :3], img, two_chan_trimap), 1)

        output = self.conv_up4(x)

        alpha = torch.clamp(output[:, 0][:, None], 0, 1) # clip(alpha, 0, 1)
        F = torch.sigmoid(output[:, 1:4]) 
        B = torch.sigmoid(output[:, 4:7])

        # FBA Fusion
        alpha, F, B = self.fba_fusion(alpha, img, F, B)

        output = torch.cat((alpha, F, B), 1)

        return output


    def fba_fusion(self, alpha, img, F, B):
        F = ((alpha * img + (1 - alpha**2) * F - alpha * (1 - alpha) * B))
        B = ((1 - alpha) * img + (2 * alpha - alpha**2) * B - alpha * (1 - alpha) * F)

        F = torch.clamp(F, 0, 1)
        B = torch.clamp(B, 0, 1)
        la = 0.1
        alpha = (alpha * la + torch.sum((img - B) * (F - B), 1, keepdim=True)) / (torch.sum((F - B) * (F - B), 1, keepdim=True) + la)
        alpha = torch.clamp(alpha, 0, 1)
        return alpha, F, B

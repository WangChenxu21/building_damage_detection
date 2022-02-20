import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ConvRelu, SCSEModule
from .backbones.dpn import dpn92


class Dpn_Loc(nn.Module):
    def __init__(self, pretrained='imagenet+5k', **kwargs):
        super(Dpn_Loc, self).__init__()
        
        encoder_filters = [64, 336, 704, 1552, 2688]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = nn.Sequential(ConvRelu(decoder_filters[-1]+encoder_filters[-2], decoder_filters[-1]), SCSEModule(decoder_filters[-1], reduction=16, concat=True))
        self.conv7 = ConvRelu(decoder_filters[-1] * 2, decoder_filters[-2])
        self.conv7_2 = nn.Sequential(ConvRelu(decoder_filters[-2]+encoder_filters[-3], decoder_filters[-2]), SCSEModule(decoder_filters[-2], reduction=16, concat=True))
        self.conv8 = ConvRelu(decoder_filters[-2] * 2, decoder_filters[-3])
        self.conv8_2 = nn.Sequential(ConvRelu(decoder_filters[-3]+encoder_filters[-4], decoder_filters[-3]), SCSEModule(decoder_filters[-3], reduction=16, concat=True))
        self.conv9 = ConvRelu(decoder_filters[-3] * 2, decoder_filters[-4])
        self.conv9_2 = nn.Sequential(ConvRelu(decoder_filters[-4]+encoder_filters[-5], decoder_filters[-4]), SCSEModule(decoder_filters[-4], reduction=16, concat=True))
        self.conv10 = ConvRelu(decoder_filters[-4] * 2, decoder_filters[-5])

        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)
        
        self._initialize_weights()

        encoder = dpn92(pretrained=pretrained)
        
        self.conv1 = nn.Sequential(
                encoder.blocks['conv1_1'].conv,  # conv
                encoder.blocks['conv1_1'].bn,  # bn
                encoder.blocks['conv1_1'].act,  # relu
            )
        self.conv2 = nn.Sequential(
                encoder.blocks['conv1_1'].pool,  # maxpool
                *[b for k, b in encoder.blocks.items() if k.startswith('conv2_')]
            )
        self.conv3 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv3_')])
        self.conv4 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv4_')])
        self.conv5 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv5_')])

    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        enc1 = (torch.cat(enc1, dim=1) if isinstance(enc1, tuple) else enc1)
        enc2 = (torch.cat(enc2, dim=1) if isinstance(enc2, tuple) else enc2)
        enc3 = (torch.cat(enc3, dim=1) if isinstance(enc3, tuple) else enc3)
        enc4 = (torch.cat(enc4, dim=1) if isinstance(enc4, tuple) else enc4)
        enc5 = (torch.cat(enc5, dim=1) if isinstance(enc5, tuple) else enc5)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return self.res(dec10)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Dpn_Double(nn.Module):
    def __init__(self, pretrained='imagenet+5k', **kwargs):
        super(Dpn_Double, self).__init__()
        
        encoder_filters = [64, 336, 704, 1552, 2688]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = nn.Sequential(ConvRelu(decoder_filters[-1]+encoder_filters[-2], decoder_filters[-1]), SCSEModule(decoder_filters[-1], reduction=16, concat=True))
        self.conv7 = ConvRelu(decoder_filters[-1] * 2, decoder_filters[-2])
        self.conv7_2 = nn.Sequential(ConvRelu(decoder_filters[-2]+encoder_filters[-3], decoder_filters[-2]), SCSEModule(decoder_filters[-2], reduction=16, concat=True))
        self.conv8 = ConvRelu(decoder_filters[-2] * 2, decoder_filters[-3])
        self.conv8_2 = nn.Sequential(ConvRelu(decoder_filters[-3]+encoder_filters[-4], decoder_filters[-3]), SCSEModule(decoder_filters[-3], reduction=16, concat=True))
        self.conv9 = ConvRelu(decoder_filters[-3] * 2, decoder_filters[-4])
        self.conv9_2 = nn.Sequential(ConvRelu(decoder_filters[-4]+encoder_filters[-5], decoder_filters[-4]), SCSEModule(decoder_filters[-4], reduction=16, concat=True))
        self.conv10 = ConvRelu(decoder_filters[-4] * 2, decoder_filters[-5])

        self.res = nn.Conv2d(decoder_filters[-5] * 2, 5, 1, stride=1, padding=0)
        
        self._initialize_weights()

        encoder = dpn92(pretrained=pretrained)
        
        self.conv1 = nn.Sequential(
                encoder.blocks['conv1_1'].conv,  # conv
                encoder.blocks['conv1_1'].bn,  # bn
                encoder.blocks['conv1_1'].act,  # relu
            )
        self.conv2 = nn.Sequential(
                encoder.blocks['conv1_1'].pool,  # maxpool
                *[b for k, b in encoder.blocks.items() if k.startswith('conv2_')]
            )
        self.conv3 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv3_')])
        self.conv4 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv4_')])
        self.conv5 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv5_')])

    def forward1(self, x):
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        enc1 = (torch.cat(enc1, dim=1) if isinstance(enc1, tuple) else enc1)
        enc2 = (torch.cat(enc2, dim=1) if isinstance(enc2, tuple) else enc2)
        enc3 = (torch.cat(enc3, dim=1) if isinstance(enc3, tuple) else enc3)
        enc4 = (torch.cat(enc4, dim=1) if isinstance(enc4, tuple) else enc4)
        enc5 = (torch.cat(enc5, dim=1) if isinstance(enc5, tuple) else enc5)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10

    def forward(self, x):
        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])

        dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

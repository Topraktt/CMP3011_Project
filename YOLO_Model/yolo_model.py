import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1) if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPLayer, self).__init__()
        self.split_conv = ConvModule(in_channels, out_channels // 2, 1, 1, 0)
        self.blocks = nn.Sequential(
            *[ConvModule(out_channels // 2, out_channels // 2, 3, 1, 1) for _ in range(num_blocks)]
        )
        self.concat_conv = ConvModule(out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        y1 = self.split_conv(x)
        y2 = self.blocks(y1)
        return self.concat_conv(torch.cat([y1, y2], dim=1))

class SPFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPFF, self).__init__()
        self.conv1 = ConvModule(in_channels, in_channels // 2, 1, 1, 0)
        self.conv2 = ConvModule(in_channels // 2, out_channels, 3, 1, 1)
        self.pooling = nn.ModuleList([ 
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=9, stride=1, padding=4),
            nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        ])
        self.concat_conv = ConvModule(in_channels // 2 * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        pooled = [pool(x1) for pool in self.pooling]
        return self.concat_conv(torch.cat([x1, *pooled], dim=1))

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.stem = ConvModule(3, 32, 3, 1, 1)
        self.stage1 = CSPLayer(32, 64, 1)
        self.stage2 = CSPLayer(64, 128, 3)
        self.stage3 = CSPLayer(128, 256, 3)
        self.stage4 = CSPLayer(256, 512, 1)
        self.spff = SPFF(512, 512)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.spff(x4)
        return x1, x2, x3, x4, x5

class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        self.topdown1 = CSPLayer(512, 256, 1)
        self.topdown2 = CSPLayer(256, 128, 1)
        self.bottomup1 = CSPLayer(128, 256, 1)
        self.bottomup2 = CSPLayer(256, 512, 1)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        t1 = self.topdown1(F.interpolate(x5, scale_factor=2, mode='nearest') + x4)
        t2 = self.topdown2(F.interpolate(t1, scale_factor=2, mode='nearest') + x3)
        b1 = self.bottomup1(F.interpolate(t2, scale_factor=0.5, mode='nearest') + x2)
        b2 = self.bottomup2(F.interpolate(b1, scale_factor=0.5, mode='nearest') + x1)
        return t2, b1, b2

class Head(nn.Module):
    def __init__(self, num_classes):
        super(Head, self).__init__()
        self.detect1 = nn.Conv2d(128, num_classes, 1)
        self.detect2 = nn.Conv2d(256, num_classes, 1)
        self.detect3 = nn.Conv2d(512, num_classes, 1)

    def forward(self, features):
        t2, b1, b2 = features
        return self.detect1(t2), self.detect2(b1), self.detect3(b2)

class YOLOv8(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8, self).__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(num_classes)

    def forward(self, x):
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        return self.head(neck_features)

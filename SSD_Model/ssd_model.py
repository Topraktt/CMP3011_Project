import torch
import torch.nn as nn
import torchvision


class SSDLite(nn.Module):
    def __init__(self, num_classes=3):
        super(SSDLite, self).__init__()

        mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
        self.backbone = nn.Sequential(*list(mobilenet.features[:14]))

        self.extras = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
        ])

        self.loc_layers = nn.ModuleList([
            nn.Conv2d(96, 24, kernel_size=3, padding=1),
            nn.Conv2d(512, 24, kernel_size=3, padding=1),
            nn.Conv2d(256, 24, kernel_size=3, padding=1),
            nn.Conv2d(256, 24, kernel_size=3, padding=1)
        ])

        self.conf_layers = nn.ModuleList([
            nn.Conv2d(96, num_classes * 6, kernel_size=3, padding=1),
            nn.Conv2d(512, num_classes * 6, kernel_size=3, padding=1),
            nn.Conv2d(256, num_classes * 6, kernel_size=3, padding=1),
            nn.Conv2d(256, num_classes * 6, kernel_size=3, padding=1)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features = []
        x = self.backbone(x)
        features.append(x)

        for i, layer in enumerate(self.extras):
            x = layer(x)
            features.append(x)

        loc_preds = []
        conf_preds = []
        for i, feat in enumerate(features):
            loc_preds.append(self.loc_layers[i](feat).permute(0, 2, 3, 1).contiguous())
            conf_preds.append(self.conf_layers[i](feat).permute(0, 2, 3, 1).contiguous())

        loc_preds = torch.cat([o.view(o.size(0), -1) for o in loc_preds], 1)
        conf_preds = torch.cat([o.view(o.size(0), -1) for o in conf_preds], 1)

        return loc_preds, conf_preds

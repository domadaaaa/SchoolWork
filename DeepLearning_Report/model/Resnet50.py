import torch
from torch import nn
import torch.nn.functional as func

expansion = 4  # bottleneck的扩展系数


class ResidualBlock(nn.Module):  # 残差块
    def __init__(self, in_c, mid_c, out_c, downsample=False, strides=1):  # stride调整hw进行shortcut
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, mid_c, kernel_size=1, stride=1)  # 1×1
        self.conv2 = nn.Conv2d(mid_c, mid_c, kernel_size=3, stride=strides, padding=1)  # 3×3
        self.conv3 = nn.Conv2d(mid_c, out_c, kernel_size=1, stride=1)  # 1×1
        self.conv_c = nn.Conv2d(in_c, out_c, kernel_size=1, stride=strides)  # 用于shortcut的1×1
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn1(self.conv2(y)))
        if self.downsample:  # 输入输出维度不一致
            x = self.conv_c(x)
            y = self.bn2(self.conv3(y))
            return self.relu(y+x)
        else:  # 输入输出维度一致
            y = self.bn2(self.conv3(y))
            return self.relu(y+x)


class Res50(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, num_class=2):
        super(Res50, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.residual = nn.Sequential(
            # stage 1
            ResidualBlock(64, 64, 256, downsample=True), ResidualBlock(256, 64, 256), ResidualBlock(256, 64, 256),
            # stage 2
            ResidualBlock(256, 128, 512, downsample=True, strides=2), ResidualBlock(512, 128, 512), ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            # stage 3
            ResidualBlock(512, 256, 1024, downsample=True, strides=2), ResidualBlock(1024, 256, 1024), ResidualBlock(1024, 256, 1024),
            ResidualBlock(1024, 256, 1024), ResidualBlock(1024, 256, 1024), ResidualBlock(1024, 256, 1024),
            # stage 4
            ResidualBlock(1024, 512, 2048, downsample=True, strides=2), ResidualBlock(2048, 512, 2048), ResidualBlock(2048, 512, 2048),
        )
        self.aft = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_class, bias=True),
            nn.Softmax()
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 将权重调整为 均值为0 方差为0.01 的正态分布函数
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pre(x)
        x = self.residual(x)
        x = self.aft(x)
        return x

import torch
from torch import nn


def _make_divisible(ch, divisor=8, min_ch=None):  # 将输出通道数调整为divisor的整数倍，min_ch为最小通道数
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)  # (ch + divisor / 2) // divisor 是用来四舍五入的
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# ConvBNReLU即一个层
class ConvBNReLU(nn.Sequential):  # 继承nn.Sequential不需要写forward函数
    # group意思是将对应的输入通道与输出通道数进行分组, 默认值为1, 也就是说默认输出输入的所有通道各为一组。
    # 比如输入数据大小为90x100x100x32，通道数32，要经过一个3x3x48的卷积，group默认是1，就是全连接的卷积层。
    # 如果group是2，那么对应要将输入的32个通道分成2个16的通道，将输出的48个通道分成2个24的通道。
    # 对输出的2个24的通道，第一个24通道与输入的第一个16通道进行全卷积，第二个24通道与输入的第二个16通道进行全卷积。
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):  # group=1是普通卷积，group=inchannel是dw卷积
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),  # 采用BN，偏置不起作用？
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)  # min(max(x,0),6)
        )


class InvertedResidual(nn.Module):  # Inverted Bottleneck
    def __init__(self, in_channel, out_channel, stride, expand_ratio):  # expand为扩展因子
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio  # bneck中间层的通道数
        self.use_shortcut = stride == 1 and in_channel == out_channel  # 判断是否使用shortcut

        layers = []  # 定义层列表
        if expand_ratio != 1:  # 扩展因子不等于1才有第一层的1×1卷积层
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)  # *layers位置参数

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2, alpha=1.0, round_nearest=8):  # alpha控制卷积层所用卷积核个数，将输出通道数调整为round_nearest的整数倍
        super(MobileNetV2, self).__init__()
        block = InvertedResidual  # block即一个倒残差结构
        input_channel = _make_divisible(32 * alpha, round_nearest)  # 定义初始输入通道
        last_channel = _make_divisible(1280 * alpha, round_nearest)  # 定义最后输出通道
        # 对应每一层的参数 t代表每一个bneck中间维度都扩大了6倍
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        features = []  # 特征提取层列表
        # conv1 layer
        features.append(ConvBNReLU(in_channel=3, out_channel=input_channel, stride=2))  # kernelsize默认值为3
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)  # 利用c，计算每一个block应得的output_channel
            for i in range(n):
                stride = s if i == 0 else 1  # 利用n，对应bottleneck的第一层stride=s，否则都为1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))  # 扩展因子即t，计算hidden_channel
                input_channel = output_channel  # 每一层结束后输出通道作为下一层的输入通道
        # building last several layers 逐点卷积(PW)层
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均下采样
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

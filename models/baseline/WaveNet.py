import torch
import torch.nn as nn
from torchinfo import summary

"""
两种实现方式
1. Graph WaveNet
    12个step先pad到13 经过一堆conv layer之后正好缩成1个step
    也就是把时间维卷没了
    最后把channel维映射成12个out step
2. 原版 WaveNet
    每层conv输入输出尺寸保持一致
    每次conv前都要pad
    
这里按照gwnet的版本实现
"""


class CasualConv(nn.Module):
    """
    out_length = [in_length + 2*pad - dilation*(kernel_size-1) - 1] / stride + 1

    if stride == 1:
        out_length = in_length + 2*pad - dilation*(kernel_size-1)
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            dilation=dilation,
            stride=stride,
        )

    def forward(self, x):
        # x: (B, C, N, T)

        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, hidden_channels, kernel_size=2, num_layers=2):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dilation = 1

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for _ in range(num_layers):
            self.filter_convs.append(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=(1, kernel_size),
                    dilation=self.dilation,
                )
            )

            self.gate_convs.append(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=(1, kernel_size),
                    dilation=self.dilation,
                )
            )

            self.residual_convs.append(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=(1, 1),
                )
            )

            self.skip_convs.append(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=(1, 1),
                )
            )

            self.dilation *= 2

    def forward(self, x):
        # x: (B, C, N, T)

        skip_list = []

        for i in range(self.num_layers):
            residual = x

            filter = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filter * gate

            skip = self.skip_convs[i](x)
            skip_list.append(skip)

            x = self.residual_convs[i](x)
            x = (
                x + residual[:, :, :, -x.shape[3] :]
            )  # truncate because x get shorter after conv

        return x, skip_list


class WaveNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=12,
        hidden_channels=16,
        kernel_size=2,
        num_blocks=4,
        num_layers=2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers

        self.receptive_field = 1
        for _ in range(num_blocks):
            additional_scope = kernel_size - 1
            for _ in range(num_layers):
                self.receptive_field += additional_scope
                additional_scope *= 2

        self.input_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1))
        self.blocks = nn.ModuleList(
            ConvBlock(hidden_channels, kernel_size, num_layers)
            for _ in range(num_blocks)
        )
        self.output_proj = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=(1, 1)),
        )

    def forward(self, x):
        # x: (B, T, N, C)
        in_steps = x.shape[1]

        x = x.permute(0, 3, 2, 1)  # (B, C, N, T)
        if in_steps < self.receptive_field:
            x = nn.functional.pad(
                x, (self.receptive_field - in_steps, 0, 0, 0)
            )  # param: (pad_left, pad_right, pad_top, pad_bottom)
        elif in_steps > self.receptive_field:
            print(f"WARNING: using only last {self.receptive_field} input steps!")
            x = x[..., -self.receptive_field :]

        x = self.input_conv(x)  # (B, hidden_channels, N, T+?)

        skips = []
        for block in self.blocks:
            x, skip_list = block(x)
            skips.extend(skip_list)

        # all skip conn will be truncated to final_length
        final_length = skips[-1].shape[3]
        skip_sum = skips[0][..., -final_length:]
        for i in range(1, len(skips)):
            skip_sum += skips[i][
                ..., -final_length:
            ]  # (B, hidden_channels, N, final_length)

        out = self.output_proj(skip_sum)  # (B, out_channels, N, final_length)
        # e.g. (B, 12, N, 1)

        return out


if __name__ == "__main__":
    model = WaveNet()
    summary(model, [64, 12, 207, 1])

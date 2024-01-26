import torch
import torch.nn as nn
import torch.nn.functional as F

global CONV, CONVTRANSP, BN, POOL
CONV = nn.Conv2d
CONVTRANSP = nn.ConvTranspose2d
BN = nn.BatchNorm2d
POOL = nn.MaxPool2d


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            CONV(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            BN(F_int),
        )

        self.W_x = nn.Sequential(
            CONV(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            BN(F_int),
        )

        self.psi = nn.Sequential(
            CONV(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            BN(1),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        alpha = F.relu(g1 + x1)
        alpha = self.psi(alpha)
        return x * alpha


class DoubleConv(nn.Module):
    def __init__(self, channels, activation=nn.ReLU(), residual=False):
        super().__init__()
        self.conv = nn.Sequential(
            CONV(channels[0], channels[1], kernel_size=3, padding=1),
            BN(channels[1]),
            activation,
            CONV(channels[1], channels[2], kernel_size=3, padding=1),
            BN(channels[2]),
            activation,
        )

        self.use_residual = residual
        if self.use_residual:
            self.residual = nn.Sequential(
                CONV(channels[0], channels[2], kernel_size=1, padding=0),
                BN(channels[2]),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + self.residual(x)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), residual=False):
        super().__init__()
        self.pool = POOL(2)
        self.conv = DoubleConv(
            [in_channels, in_channels, out_channels], activation, residual
        )

    def forward(self, x):
        skip = self.conv(x)
        out = self.pool(skip)
        return out, skip


class UpBlock(nn.Module):
    def __init__(
        self, skip_channels, in_channels, activation=nn.ReLU(), residual=False
    ):
        super().__init__()
        self.up = CONVTRANSP(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(
            [skip_channels + in_channels, skip_channels, skip_channels],
            activation,
            residual,
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        init_features=32,
        depth=3,
        activation=nn.ReLU(),
        residual=False,
        attention=False,
        dimension=2,
    ):
        super().__init__()

        assert dimension in [2, 3], f"dimension={dimension} should be in [2, 3]"
        global CONV, CONVTRANSP, BN, POOL
        if dimension == 2:
            CONV = nn.Conv2d
            BN = nn.BatchNorm2d
            POOL = nn.MaxPool2d
            CONVTRANSP = nn.ConvTranspose2d
        else:
            CONV = nn.Conv3d
            BN = nn.BatchNorm3d
            POOL = nn.MaxPool3d
            CONVTRANSP = nn.ConvTranspose3d

        self.dimension = dimension
        self.depth = depth
        self.residual = residual
        self.attention = attention

        self.encoder = nn.ModuleList()

        for i in range(depth):
            if i == 0:
                self.encoder.append(
                    DownBlock(
                        in_channels,
                        init_features,
                        activation=activation,
                        residual=residual,
                    )
                )
            else:
                self.encoder.append(
                    DownBlock(
                        init_features * 2 ** (i - 1),
                        init_features * 2**i,
                        activation=activation,
                        residual=residual,
                    )
                )

        self.bottleneck = DoubleConv(
            [
                init_features * 2 ** (depth - 1),
                init_features * 2 ** (depth - 1),
                init_features * 2**depth,
            ],
            activation,
        )

        self.decoder = nn.ModuleList()

        for i in reversed(range(depth)):
            self.decoder.append(
                UpBlock(
                    in_channels=init_features * 2 ** (i + 1),
                    skip_channels=init_features * 2**i,
                    activation=activation,
                    residual=residual,
                )
            )

        if self.attention:
            self.attention_upsample = nn.Upsample(scale_factor=2)
            self.attention_gates = nn.ModuleList()
            for i in reversed(range(depth)):
                self.attention_gates.append(
                    AttentionGate(
                        F_g=init_features * 2 ** (i + 1),
                        F_l=init_features * 2**i,
                        F_int=init_features * 2**i,
                    )
                )

        self.out = CONV(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x, skip = self.encoder[i](x)
            skips.append(skip)
        x = self.bottleneck(x)
        for i in range(self.depth):
            if self.attention:
                skips[-i - 1] = self.attention_gates[i](
                    g=self.attention_upsample(x), x=skips[-i - 1]
                )
                x = self.decoder[i](x, skips[-i - 1])
            else:
                x = self.decoder[i](x, skips[-i - 1])
        x = self.out(x)
        return x

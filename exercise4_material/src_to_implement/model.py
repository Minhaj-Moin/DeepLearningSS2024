import torch


class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(3, 2)
        self.resblock1 = ResBlock(64, 64, 1)
        self.resblock2 = ResBlock(64, 128, 2)
        self.resblock3 = ResBlock(128, 256, 2)
        self.resblock4 = ResBlock(256, 512, 2)
        self.avg_pool = torch.nn.AvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn(self.conv(x))))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        x = x.mean([2, 3])
        x = self.fc(x)
        out = self.sigmoid(x)

        return out


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.res_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.out_channels, 3, stride=self.stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.out_channels)
        )

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = False
        if self.stride != 1 or self.in_channels != self.out_channels:
            self.downsample = True

        self.conv1x1 = torch.nn.Conv2d(self.in_channels, self.out_channels, 1, stride=self.stride, padding=0,
                                       bias=False)
        self.bn2 = torch.nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        identity = x
        if self.downsample: identity = self.bn2(self.conv1x1(x))
        return self.relu(self.res_block(x) + identity)

from torch import nn
from torchvision.models.optical_flow.raft import ResidualBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.batchnorm1 = nn.BatchNorm2d()
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.globalavgpool = nn.AdaptiveAvgPool2d()
        self.FC = nn.Linear(in_features=512, out_features=2)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.res1 = ResidualBlock(64, 64,1)
        self.res2 = ResidualBlock(64,128,2)
        self.res3 = ResidualBlock(128, 256,2)
        self.res4 = ResidualBlock(256, 512,2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.globalavgpool(x)
        x = self.flatten(x)
        x = self.FC(x)
        x = self.sigmoid(x)
        return x


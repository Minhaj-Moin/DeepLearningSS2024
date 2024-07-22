import torch
import numpy as np

class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ImgProcessBlock = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 5, 2, 0, bias=False),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(3, 2),
        ResBlock(64, 64, 1),
        ResBlock(64, 128, 2),
        ResBlock(128, 256, 2),
        ResBlock(256, 512, 2))
        self.FCLayer = torch.nn.Linear(512, 2)
        self.Sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.ImgProcessBlock(x)
        x = x.mean([2, 3]) #np.arange(len(x.shape))[2:].tolist() | take mean over spatial axes
        x = self.FCLayer(x)
        return self.Sigmoid(x)


class ResBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, stride):
        super(ResBlock, self).__init__()

        self.ResBlock_ = torch.nn.Sequential(
            torch.nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), # No Stride for 2nd Conv
            torch.nn.BatchNorm2d(out_c))

        self.ReLU = torch.nn.ReLU(inplace=True)
        self.downsample = (stride != 1) or (in_c != out_c)
        self.Conv = torch.nn.Conv2d(in_c, out_c, 1, stride=stride, padding=0,bias=False)
        self.BatchNorm = torch.nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.ReLU(self.ResBlock_(x) + self.BatchNorm(self.Conv(x)) if self.downsample else x)

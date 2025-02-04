import torch
from torch import nn

INPUT_SHAPE = 3

class BigELFVisionNN(nn.Module):
    def __init__(self, output_shape: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
                nn.Conv2d(in_channels=INPUT_SHAPE,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
                nn.Conv2d(in_channels=64,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_3 = nn.Sequential(
                nn.Conv2d(in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU()
        )

        self.conv_block_4 = nn.Sequential(
                nn.Conv2d(in_channels=256,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 7 * 7,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        # x = self.conv_block_0(x)
        # print(x.shape)
        x = self.conv_block_1(x)
        print(x.shape)
        x = self.conv_block_2(x)
        print(x.shape)
        x = self.conv_block_3(x)
        print(x.shape)
        x = self.conv_block_4(x)
        print(x.shape)
        x = self.classifier(x)
        return x

class ELFVisionNN(nn.Module):
    def __init__(self, output_shape: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_SHAPE, 
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, 
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 14 * 14,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.conv_block_3(x)
        # print(x.shape)
        x = self.classifier(x)
        return x

# import lubraries
import torch
import torch.nn as nn
import numpy

"""
Information about architecture
Tuple is structured by (kernal_size, filters, stride, padding)
"M" is maxpooling with stride 2 and kernel size 2
"""

# architecture configuration
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]
"""
Convolitional block
"""

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        # convolitional layer
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # batch normalization
        self.batchnorm = nn.BatchNorm2d(out_channels)
        # activation function
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        # forward pass
        return self.leaky_relu(self.batchnorm(self.conv(x)))

class YoloV1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        # call super class
        super(YoloV1, self).__init__()
        self.architecture = architecture_config
        # assign number of channels
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        # i will put all layers in this list
        layers = []
        in_channels = self.in_channels

        # iterate over architecture
        for x in architecture:
            # if element is tuple then it is conv layer
            if type(x) == tuple:
                # unpack tuble and assign values
                layers += [
                    # create CNN  block witrh given parameters
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]

                # change number of channels
                in_channels = x[1]

            # if element is string then it is maxpooling layer
            if type(x) == str:
                # add maxpooling layer
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # if element is list then it is list of conv layers
            if type(x) == list:
                # unpck list
                # first conv layer
                conv1 = x[0]
                # second conv layer
                conv2 = x[1]
                # number of times we need to repeat this block
                num_repeats = x[2]

                for _ in range(num_repeats):
                    # add layers N times
                    layers += [
                        CNNBlock(in_channels, conv1[1], kernel_size=conv1[0],
                                 stride=conv1[2], padding=conv1[3]),]
                    layers += [
                        CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], padding=conv2[3])
                    ]
                    # assign number of channels
                    in_channels = conv2[1]
        # convert list to sequential
        return nn.Sequential(*layers)

    def _create_fcs(self, S, B, C):
        """
        Method creates fully connected layers
        S is number of grid cells
        B is number of bounding boxes
        C is number of classes
        """
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )



#test()

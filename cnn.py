import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x7x7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x7x7
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x3x3
        )

        self.dense = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),  # fc4 64*3*3 -> 128
            nn.ReLU(),
            nn.Linear(128, 500)  # fc5 128->10
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)  # 64x3x3
        res = conv3_out.view(conv3_out.size(0), -1)  # batch x (64*3*3)
        out = self.dense(res)
        return out


class Net112(nn.Module):
    def __init__(self):
        super(Net112, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32x112x112
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x56x56
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x28x28
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x28x28
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x14x14
        )

        self.dense = nn.Sequential(
            nn.Linear(64 * 14 * 14, 128),  # fc4 64*14*14 -> 128
            nn.ReLU(),
            nn.Linear(128, 4)  # fc5 128->4
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)  # 64x3x3
        res = conv3_out.view(conv3_out.size(0), -1)  # batch x (64*3*3)
        out = self.dense(res)
        return out


class Net224(nn.Module):
    def __init__(self):
        super(Net224, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32x224x224
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32x112x112
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x112x112
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x56x56
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x56x56
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x28x28
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 96, 3, 1, 1),  # 96x28x28
            nn.ReLU(),
            nn.MaxPool2d(2)  # 96x14x14
        )

        self.dense = nn.Sequential(
            nn.Linear(96 * 14 * 14, 128),  # fc4 64*3*3 -> 128
            nn.ReLU(),
            nn.Linear(128, 4)  # fc5 128->4
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        res = conv4_out.view(conv4_out.size(0), -1)
        out = self.dense(res)
        return out


class Net96(nn.Module):
    def __init__(self):
        super(Net96, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32x96x96
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32x48x48
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x48x48
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x24x24
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x24x24
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x12x12
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 96, 3, 1, 1),  # 96x12x12
            nn.ReLU(),
            nn.MaxPool2d(2)  # 96x6x6
        )

        self.dense = nn.Sequential(
            nn.Linear(96 * 6 * 6, 128),  # fc4 96*6*6 -> 128
            nn.ReLU(),
            nn.Linear(128, 4)  # fc5 128->4
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        res = conv4_out.view(conv4_out.size(0), -1)
        out = self.dense(res)
        return out


class Net64x48(nn.Module):
    def __init__(self):
        super(Net64x48, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32x64x48
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x32x24
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x16x12
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x16x12
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x8x6
        )

        self.dense = nn.Sequential(
            nn.Linear(64 * 8 * 6, 128),  # fc4 64*8*6 -> 128
            nn.ReLU(),
            nn.Linear(128, 2)  # fc5 128->2
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out


class Net_fcn(nn.Module):
    def __init__(self):
        super(Net_fcn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x7x7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x7x7
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x3x3
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 10, 3, 1, 0)  # 64x3x3
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)  # 64x3x3
        out = self.conv4(conv3_out)
        return out


class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.model = nn.Sequential(

            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 120, 120]  output[48, 55, 55]

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]

            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]

            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 100),

        )

    def forward(self, x):
        x = self.model(x)
        return x

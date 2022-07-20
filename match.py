import torch
import torch.nn as nn
from torchvision import transforms

# from models.cnn import Net
from PIL import Image
import numpy as np

data_transform = transforms.Compose(
    [transforms.Resize(28),
     transforms.ToTensor()])


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
            nn.Linear(128, 214)  # fc5 128->10
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)  # 64x3x3
        res = conv3_out.view(conv3_out.size(0), -1)  # batch x (64*3*3)
        out = self.dense(res)
        return out


def resnet_cifar(net, input_data):
    x = net.conv1(input_data)
    x = net.conv2(x)
    x = net.conv3(x)
    return x


def Euclidean(a, a1):
    l = np.linalg.norm(a - a1)
    return l


def eucliDist(A, B):
    return np.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


def Manhattan(a, a1):
    l = sum(abs(a - a1))
    return l


def feature():
    model = Net()
    # model = models.resnet18(num_classes=10)  # 调用内置模型
    # model = models.resnet101(num_classes=10)  # 调用内置模型
    # model.load_state_dict(torch.load('./output/params_10.pth'))
    # from torchsummary import summary
    # summary(model, (3, 28, 28))

    model_weight_path = "/Users/xiaomo/PycharmProjects/final/params_40.pth"
    model.load_state_dict(torch.load(model_weight_path))
    f = open("4.txt", "r")
    lines = f.readlines()  # 读取全部内容
    for line in lines:
        j = line.rstrip('\n')
        if int(j) > 3493: continue
        path = "/Users/xiaomo/PycharmProjects/data/4/" + str(j) + "/"
        img = Image.open("/Users/xiaomo/PycharmProjects/final/standard/" + str(j) + ".png")
        img = data_transform(img)
        x = resnet_cifar(model, img)
        a = x.detach().numpy()
        b = a.ravel()
        dismax = 0
        for i in range(1, 1001):
            img1 = Image.open(path + str(j) + '_' + str(i) + ".png")
            img1 = data_transform(img1)
            x1 = resnet_cifar(model, img1)
            a1 = x1.detach().numpy()
            b1 = a1.ravel()

            dis1 = eucliDist(b, b1)

            dismax = max(dismax, dis1)

        g = open("no.txt", "r")
        glines = g.readlines()  # 读取全部内容
        for line in glines:
            x = line.rstrip('\n')
            img2 = Image.open(str(x))
            img2 = data_transform(img2)
            x2 = resnet_cifar(model, img2)
            a2 = x2.detach().numpy()
            b2 = a2.ravel()

            dis2 = eucliDist(b, b2)

            if dis2 - dismax < -50:
                print(x)
                print(j)


if __name__ == '__main__':
    feature()

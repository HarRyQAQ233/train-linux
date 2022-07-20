import torch
import torch.nn as nn
from torchvision import transforms

# from models.cnn import Net
from PIL import Image
import numpy as np

data_transform = transforms.Compose(
    [transforms.Resize(28),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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
            nn.Linear(128, 601)  # fc5 128->10
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)  # 64x3x3
        res = conv3_out.view(conv3_out.size(0), -1)  # batch x (64*3*3)
        out = self.dense(res)
        return out


def resnet_cifar1(net, input_data):
    x = net.conv1(input_data)
    x = net.conv2(x)
    x = net.conv3(x)
    res = x.view(x.size(0), -1)
    x = net.dense(x)
    # x = x.view(x.shape[0], -1)
    return x


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


def distEclud(arrA, arrB):
    d = arrA - arrB
    dist = np.sum(np.power(d, 2), axis=1)
    return dist


# 余弦距离
def distCos(vecA, vecB):
    # print(vecA)
    # print(vecB)
    return float(np.sum(np.array(vecA) * np.array(vecB))) / (
            distEclud(vecA, np.mat(np.zeros(len(vecA[0])))) * distEclud(vecB, np.mat(np.zeros(len(vecB[0])))))


def dot(vector1, vector2):
    cosV12 = dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cosV12


def cosine_distance(arr1, arr2):
    """
	Calculate the cosine distance between arr1 and arr2.
	:param arr1: np.array
	:param arr2: np.array
	:return:
	"""
    assert arr1.ndim == 1 and arr2.ndim == 1
    distance = np.dot(arr1, arr2) / (np.sqrt(np.sum(arr1 ** 2) * np.sum(arr2 ** 2)))
    return distance


def Manhattan(a, a1):
    l = sum(abs(a - a1))
    return l


j = 737


def feature():
    model = Net()
    # model = models.resnet18(num_classes=10)  # 调用内置模型
    # model = models.resnet101(num_classes=10)  # 调用内置模型
    # model.load_state_dict(torch.load('./output/params_10.pth'))
    # from torchsummary import summary
    # summary(model, (3, 28, 28))

    model_weight_path = "/Users/xiaomo/PycharmProjects/final/params_best.pth"
    model.load_state_dict(torch.load(model_weight_path))

    path = "/Users/xiaomo/PycharmProjects/data/4/" + str(j) + "/"
    # 打印出模型的结构
    print(model)
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
        # dis1 = distCos(a, a1)

        dismax = max(dismax, dis1)

    # f = open("no.txt", "r")
    # lines = f.readlines()  # 读取全部内容
    # for line in lines:
    #     x = line.rstrip('\n')
    x = "/Users/xiaomo/PycharmProjects/final/479no4.png"
    img2 = Image.open(str(x))
    img2 = data_transform(img2)
    x2 = resnet_cifar(model, img2)
    a2 = x2.detach().numpy()
    b2 = a2.ravel()

    dis2 = eucliDist(b, b2)
    # dis2 = distCos(a, a2)

    print(dismax)
    print(dis2)
    print(dis2-dismax)

if __name__ == '__main__':
    feature()

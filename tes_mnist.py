import torch
import cv2
from torch.autograd import Variable
from torchvision import transforms
from cnn import Net
from toonnx import to_onnx

use_cuda = False
model = Net()
model.load_state_dict(torch.load('/Users/xiaomo/PycharmProjects/final/params_40.pth'))
# model = torch.load('output/model.pth')
model.eval()
if use_cuda and torch.cuda.is_available():
    model.cuda()

to_onnx(model, 3, 28, 28, 'output/params.onnx')

img = cv2.imread('479no4.png')
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.unsqueeze(0)
if use_cuda and torch.cuda.is_available():
    prediction = model(Variable(img_tensor.cuda()))
else:
    prediction = model(Variable(img_tensor))
pred = torch.max(prediction, 1)[1]
print(pred)
# cv2.imshow("image", img)
# cv2.waitKey(0)
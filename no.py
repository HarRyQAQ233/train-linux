import os
import random
import cv2
from torch.utils.data import Dataset

f = open('no.txt', 'wt')
path = "/Users/xiaomo/PycharmProjects/final/no/"
lst = os.listdir(path)
if '.DS_Store' in lst:    lst.remove('.DS_Store')
for i in lst:
    fullname = os.path.join(path, i)
    f.write(fullname + '\n')
f.close()




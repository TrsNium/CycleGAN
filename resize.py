import cv2
import numpy as np
import os
import datetime

Xdir_ = 'download/'
Ydir_ = 'linedraw/'

save_xdir = './Xdir/'
save_ydir = './Ydir/'

def resize(base_path, save_path, size_):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i,file_name in enumerate(os.listdir(base_path)):
        img = cv2.imread(base_path + file_name, cv2.IMREAD_COLOR)

        height = int(img.shape[0])
        width = int(img.shape[1])
        if (height < size_) or (width < size_) :
            continue

        size = (int(height * size_/height), int(width * size_/width))
        resizedImg = cv2.resize(img, size)
        cv2.imwrite(save_path+ file_name,resizedImg)


resize(Xdir_, save_xdir, 256)
resize(Ydir_, save_ydir, 256)

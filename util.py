import tensorflow as tf
from PIL import Image
import numpy as np
import random
import os

def sample(size, channel, path, batch_size):
    choice_file_names = random.sample(os.listdir(path), batch_size)
    imgs = np.empty((0,size,size,channel), dtype=np.float32)

    for file_name in choice_file_names:
        img = np.array(Image.open(path+file_name)).astype(np.float32)
        imgs = np.append(imgs, np.array([img]), axis=0)
    imgs = imgs.reshape((-1,size,size,channel))
    return imgs

def sample_X_Y(size, channel, X_path, Y_path, batch_size):
    batch_X = sample(size, channel, X_path, batch_size)
    batch_Y = sample(size, channel, Y_path, batch_size)
    return batch_X, batch_Y

def visualize(size,  X, fakeX, Y, fakeY, batch_size, i):
    for n in range(batch_size):
        img = np.concatenate((X[n], fakeY[n], Y[n], fakeX[n]),axis=1)
        img = Image.fromarray(np.uint8(img))
        img.save('./visualized/batch_num{}itr{}.jpg'.format(n,i))

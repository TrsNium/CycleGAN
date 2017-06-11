import tensorflow as tf
import numpy as np
import random
import os
import scipy.misc

def sample(size, channel, path, batch_size):
    choice_file_names = random.sample(os.listdir(path), batch_size)
    imgs = np.empty((0,size,size,channel), dtype=np.float32)
    encode = lambda x: x/127.5 -1

    for file_name in choice_file_names:
        img = encode(scipy.misc.imread(path+file_name).astype(np.float32))
        imgs = np.append(imgs, np.array([img]), axis=0)
    imgs = imgs.reshape((-1,size,size,channel))
    return imgs

def sample_X_Y(size, channel, X_path, Y_path, batch_size):
    batch_X = sample(size, channel, X_path, batch_size)
    batch_Y = sample(size, channel, Y_path, batch_size)
    return batch_X, batch_Y

def visualize(size,  X, fakeX, Y, fakeY, batch_size, i):
    decode = lambda x: (x+1.)/2
    for n in range(batch_size):
        img = np.concatenate((decode(X[n]), decode(fakeY[n]), decode(Y[n]), decode(fakeX[n])),axis=1)
        scipy.misc.imsave('./visualized/batch_num{}itr{}.jpg'.format(n,i), img)

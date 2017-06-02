import tensorflow as tf
import numpy as np
from Unet import UNet
from discriminator import Discriminator
import os
from PIL import Image
import random
import time 

class Train():
    def __init__(self):

        self.realX = tf.placeholder(tf.float32, shape=[None,512,512,3])
        self.realY = tf.placeholder(tf.float32, shape=[None,512,512,3])
        self.fakeY = UNet(self.realX, "g").dec_dc0
        self.fakeX = UNet(self.realY, "f").dec_dc0

        dis_fakey = Discriminator(self.fakeY, "dy")
        fakey_logits = dis_fakey.last_h
        fakey_out = dis_fakey.out

        dis_fakex = Discriminator(self.fakeX, "dx")
        fakex_logits = dis_fakex.last_h
        fakex_out = dis_fakex.out

        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fakey_logits, labels=tf.ones_like(fakey_out))\
                                + 100*tf.reduce_mean(tf.abs(UNet(self.fakeY, "f").dec_dc0 - self.realX))\
                                + 100*tf.reduce_mean(tf.abs(UNet(self.fakeX, "g").dec_dc0 - self.realY))
        f_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fakex_logits, labels=tf.ones_like(fakex_out))\
                                + 100*tf.reduce_mean(tf.abs(UNet(self.fakeY, "f").dec_dc0 - self.realX))\
                                + 100*tf.reduce_mean(tf.abs(UNet(self.fakeX, "g").dec_dc0 - self.realY))
        
        #discriminator
        dis_r = Discriminator(realAB, False)
        real_logits = dis_r.last_h
        real_out = dis_r.out

        dis_f = Discriminator(fakeAB, True)
        fake_logits = dis_f.last_h
        fake_out = dis_f.out

        training_var = tf.trainable_variables()
        d_var = [var for var in training_var if 'd_' in var.name]
        g_var = [var for var in training_var if 'g_' in var.name]

        self.opt_d = tf.train.AdamOptimizer(0.0002,beta1=0.5).minimize(self.d_loss, var_list=d_var)
        self.opt_g = tf.train.AdamOptimizer(0.0002,beta1=0.5).minimize(self.g_loss, var_list=g_var)

if not os.path.exists('./saved/'):
    os.mkdir('./saved/')

if not os.path.exists('./visualized/'):
    os.mkdir('./visualized/')

def sample(size, channel, path, batch_files):
    imgs = np.empty((0,size,size,channel), dtype=np.float32)

    for file_name in batch_files:
        img = np.array(Image.open(path+file_name)).astype(np.float32)
        imgs = np.append(imgs, np.array([img]), axis=0)
    imgs = imgs.reshape((-1,size,size,channel))
    return imgs

def visualize_g(size, g_img, x_img, t_img,batch_size, epoch, i):
    for n in range(batch_size):
        img = np.concatenate((g_img[n], x_img[n], t_img[n]),axis=1)
        img = Image.fromarray(np.uint8(img))
        img.save('./visualized/epoch{}batch_num{}batch{}.jpg'.format(epoch,n,i))
    

def main():
    batch_size = 5
    epochs = 200
    filenames = [random.choice(os.listdir('./data/rgb512/')) for _ in range(1000)]
    data_size = len(filenames)

    train = Train()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        graph = tf.summary.FileWriter('./logas', sess.graph)

        for epoch in range(epochs):
            new_time = time.time() 
            for i in range(0, data_size, batch_size):
                batch_files = [random.choice(filenames) for _ in range(batch_size)]
            
                rgb512 = sample(512, 3, './data/rgb512/', batch_files)
                linedraw512 = sample(512, 3, './data/linedraw512/', batch_files)
            
                batch_time = time.time()
                d_loss, _ = sess.run([train.d_loss,train.opt_d],{train.realA:rgb512,train.realB:linedraw512})  
                g_img, g_loss, _ = sess.run([train.fakeA,train.g_loss,train.opt_g],{train.realA:rgb512,train.realB:linedraw512})
             
                visualize_g(512, g_img, linedraw512, rgb512, batch_size, epoch, i)
                print('    g_loss:',g_loss,'    d_loss:',d_loss,' speed:',time.time()-batch_time," batches / s")

            print('--------------------------------')
            print('epoch_num:',epoch,'    epoch_time:',time.time()-new_time)
            print('--------------------------------')
            saver.save(sess, "saved/model.ckpt")

if __name__ == "__main__":
    main()

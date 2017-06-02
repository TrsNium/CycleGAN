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
        
        def dis_lo(img, name):
            dis = Discriminator(img, name)
            logits = dis.last_h
            out = dis.out
            return logits,out

        fakeY_logits, fakeY_out = dis_lo(self.fakeY, "dy")
        realY_logits, realY_out = dis_lo(self.realY, "dy")
        fakeX_logits, fakeX_out = dis_lo(self.fakeX, "dx")
        realX_logits, realX_out = dis_lo(self.realX, "dx")

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fakeY_logits, labels=tf.ones_like(fakeY_out)))
                                + 100*tf.reduce_mean(tf.abs(UNet(self.fakeY, "f").dec_dc0 - self.realX))
                                + 100*tf.reduce_mean(tf.abs(UNet(self.fakeX, "g").dec_dc0 - self.realY))
        self.f_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fakeX_logits, labels=tf.ones_like(fakeX_out)))
                                + 100*tf.reduce_mean(tf.abs(UNet(self.fakeY, "f").dec_dc0 - self.realX))
                                + 100*tf.reduce_mean(tf.abs(UNet(self.fakeX, "g").dec_dc0 - self.realY))
        
        self.dx_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fakeX_logits, labels=tf.zeros_like(fakeX_out)))
                                + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=realX_logits, labels=tf.ones_like(realX_out)))
        self.dy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fakeY_logits, labels=tf.zeros_like(fakeY_out)))
                                + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=realY_logits, labels=tf.ones_like(realY_out)))

        training_var = tf.trainable_variables()
        g_var = [var for var in training_var if 'g_' in var.name]
        f_var = [var for var in training_var if 'f_' in var.name]
        dx_var = [var for var in training_var if 'dx_' in var.name]
        dy_var = [var for var in training_var if 'dy_' in var.name]

        self.opt_g = tf.train.AdamOptimizer(0.0002,beta1=0.5).minimize(self.g_loss, var_list=g_var)
        self.opt_f = tf.train.AdamOptimizer(0.0002,beta1=0.5).minimize(self.f_loss, var_list=f_var)
        self.opt_dx = tf.train.AdamOptimizer(0.0002,beta1=0.5).minimize(self.dx_loss, var_list=dx_var)
        self.opt_dy = tf.train.AdamOptimizer(0.0002,beta1=0.5).minimize(self.dy_loss, var_list=dy_var)

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
                dx_loss, _ = sess.run([train.dx_loss, train.opt_dx], feed_dict={train.realX:rgb512, train.realY:linedraw512})
                dy_loss, _ = sess.run([train.dy_loss, train.opt_dy], feed_dict={train.realX:rgb512, train.realY:linedraw512})
                g_loss, _ = sess.run([train.g_loss, train.opt_g], feed_dict={train.realX:rgb512, train.realY:linedraw512})
                f_loss, _ = sess.run([train.f_loss, train.opt_f], feed_dict={train.realX:rgb512, train.realY:linedraw512})


                print('    g_loss:',g_loss,'    f_loss:',f_loss,'    dx_loss',dx_loss,'    dy_loss',dy_loss,' speed:',time.time()-batch_time," batches / s")

            print('--------------------------------')
            print('epoch_num:',epoch,'    epoch_time:',time.time()-new_time)
            print('--------------------------------')
            saver.save(sess, "saved/model.ckpt")

if __name__ == "__main__":
    main()

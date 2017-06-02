import tensorflow as tf
import numpy as np
from Unet import UNet
from discriminator import Discriminator
import os
from PIL import Image
import random
import time 
import argparse

class Train(l1_lambda, lr):

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
                                + l1_lambda*tf.reduce_mean(tf.abs(UNet(self.fakeY, "f").dec_dc0 - self.realX))
                                + l1_lambda*tf.reduce_mean(tf.abs(UNet(self.fakeX, "g").dec_dc0 - self.realY))
        self.f_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fakeX_logits, labels=tf.ones_like(fakeX_out)))
                                + l1_lambda*tf.reduce_mean(tf.abs(UNet(self.fakeY, "f").dec_dc0 - self.realX))
                                + l1_lambda*tf.reduce_mean(tf.abs(UNet(self.fakeX, "g").dec_dc0 - self.realY))
        
        self.dx_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fakeX_logits, labels=tf.zeros_like(fakeX_out)))
                                + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=realX_logits, labels=tf.ones_like(realX_out)))
        self.dy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fakeY_logits, labels=tf.zeros_like(fakeY_out)))
                                + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=realY_logits, labels=tf.ones_like(realY_out)))

        training_var = tf.trainable_variables()
        g_var = [var for var in training_var if 'g_' in var.name]
        f_var = [var for var in training_var if 'f_' in var.name]
        dx_var = [var for var in training_var if 'dx_' in var.name]
        dy_var = [var for var in training_var if 'dy_' in var.name]

        self.opt_g = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(self.g_loss, var_list=g_var)
        self.opt_f = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(self.f_loss, var_list=f_var)
        self.opt_dx = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(self.dx_loss, var_list=dx_var)
        self.opt_dy = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(self.dy_loss, var_list=dy_var)

if not os.path.exists('./saved/'):
    os.mkdir('./saved/')

if not os.path.exists('./visualized/'):
    os.mkdir('./visualized/')

def sample(size, channel, path, batch_size, names):
    choice_file_names = [random.choice(names) for _ in range(batch_size)]
    imgs = np.empty((0,size,size,channel), dtype=np.float32)

    for file_name in choice_file_names:
        img = np.array(Image.open(path+file_name)).astype(np.float32)
        imgs = np.append(imgs, np.array([img]), axis=0)
    imgs = imgs.reshape((-1,size,size,channel))
    return imgs

def visualize(size,  x, fakex, y, fakey, epoch, i):
    for n in range(batch_size):
        img = np.concatenate((x[n], fakex[n], y[n], fakey),axis=1)
        img = Image.fromarray(np.uint8(img))
        img.save('./visualized/epoch{}batch_num{}batch{}.jpg'.format(epoch,n,i))

def main(args):
    batch_size = args.batch_size
    data_size = args.data_size
    epochs = args.epochs
    x_filenames = [random.choice(os.listdir(args.xdir)) for _ in range(data_size)]
    y_filenames = [random.choice(os.listdir(args.ydir)) for _ in range(data_size)]

    train = Train(args.l1_lambda, args.lr)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        graph = tf.summary.FileWriter('./logas', sess.graph)

        for epoch in range(epochs):
            new_time = time.time() 
            for i in range(0, data_size, batch_size):                
                realX = sample(512, 3, args.xdir, batch_size, x_filenames)
                realY = sample(512, 3, args.ydir, batch_size, y_filenames)
            
                batch_time = time.time()
                dx_loss, _ = sess.run([train.dx_loss, train.opt_dx], feed_dict={train.realX:realX, train.realY:realY})
                dy_loss, _ = sess.run([train.dy_loss, train.opt_dy], feed_dict={train.realX:realX, train.realY:realY})
                fakeY, g_loss, _ = sess.run([train.fakeY, train.g_loss, train.opt_g], feed_dict={train.realX:realX, train.realY:realY})
                fakeX, f_loss, _ = sess.run([train.fakeX, train.f_loss, train.opt_f], feed_dict={train.realX:realX, train.realY:realY})

                if args.visualize:
                    visualize()

                print('    g_loss:',g_loss,'    f_loss:',f_loss,'    dx_loss',dx_loss,'    dy_loss',dy_loss,' speed:',time.time()-batch_time," batches / s")
            print('*'*8,'\n','epoch_num:',epoch,'    epoch_time:',time.time()-new_time,'*'*8,'\n')
            saver.save(sess, "saved/model.ckpt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", dest="lr", type=float, default= 0.0002)
    parser.add_argument("--xdir", dest="xdir", default="./x_data/")
    parser.add_argument("--ydir", dest="ydir", default="./y_data/")
    parser.add_argument("--epochs", dest="epochs", type=int, default=300)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=3)
    parser.add_argument("--data_size", dest="data_size", type=int, default=2000)
    parser.add_argument("--visualize", dest="visualize", type=bool, default=True)
    parser.add_argument("--l1_lambda", dest="l1_lambda", type=float, default=100.0)
    args= parser.parse_args

    main(args)

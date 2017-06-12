import tensorflow as tf
import numpy as np
import os
import argparse
from util import *

class model():

    def __init__(self, args):
        self.args = args

        self.realX = tf.placeholder(tf.float32, shape=[None,256,256,3])
        self.realY = tf.placeholder(tf.float32, shape=[None,256,256,3])
        self.fakeY = self.genereter(self.realX, "GenereterY", False)
        self.fakeX = self.genereter(self.realY, "GenereterX", False)
        fakeY_ = self.genereter(self.fakeX, "GenereterY", True)
        fakeX_ = self.genereter(self.fakeY, "GenereterX", True)

        fakeY_out = self.discriminator(self.fakeY, "DiscriminatorY", False)
        realY_out = self.discriminator(self.realY, "DiscriminatorY", True)
        fakeX_out = self.discriminator(self.fakeX, "DiscriminatorX", False)
        realX_out = self.discriminator(self.realX, "DiscriminatorX", True)

        self.g_loss = tf.reduce_mean(tf.square(fakeY_out - tf.ones_like(fakeY_out)))\
                                + self.args.l1_lambda*tf.reduce_mean(tf.abs(fakeX_ - self.realX))\
                                + self.args.l1_lambda*tf.reduce_mean(tf.abs(fakeY_ - self.realY))
        self.f_loss = tf.reduce_mean(tf.square(fakeX_out - tf.ones_like(fakeX_out)))\
                                + self.args.l1_lambda*tf.reduce_mean(tf.abs(fakeX_ - self.realX))\
                                + self.args.l1_lambda*tf.reduce_mean(tf.abs(fakeY_ - self.realY))
        
        self.dx_loss = (tf.reduce_mean(tf.square(fakeX_out - tf.zeros_like(fakeX_out)))\
                                + tf.reduce_mean(tf.square(realX_out - tf.ones_like(realX_out))))/2
        self.dy_loss = (tf.reduce_mean(tf.square(fakeY_out - tf.zeros_like(fakeY_out)))\
                                + tf.reduce_mean(tf.square(realY_out - tf.ones_like(realY_out))))/2

        training_var = tf.trainable_variables()
        self.g_var = [var for var in training_var if 'GenereterY' in var.name]
        self.f_var = [var for var in training_var if 'GenereterX' in var.name]
        self.dx_var = [var for var in training_var if 'DiscriminatorX' in var.name]
        self.dy_var = [var for var in training_var if 'DiscriminatorY' in var.name]
        
    def train(self):
        opt_g = tf.train.AdamOptimizer(self.args.lr, beta1=self.args.beta1).minimize(self.g_loss, var_list=self.g_var)
        opt_f = tf.train.AdamOptimizer(self.args.lr, beta1=self.args.beta1).minimize(self.f_loss, var_list=self.f_var)
        opt_dx = tf.train.AdamOptimizer(self.args.lr, beta1=self.args.beta1).minimize(self.dx_loss, var_list=self.dx_var)
        opt_dy = tf.train.AdamOptimizer(self.args.lr, beta1=self.args.beta1).minimize(self.dy_loss, var_list=self.dy_var)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            graph = tf.summary.FileWriter('./logs', sess.graph)

            for l in range(self.args.itrs):

                realX, realY = sample_X_Y(256, 3, self.args.Xdir, self.args.Ydir, self.args.batch_size)
                f_loss, _ = sess.run([self.f_loss, opt_f], feed_dict={self.realX:realX, self.realY:realY})
                dx_loss, _ = sess.run([self.dx_loss, opt_dx], feed_dict={self.realX:realX, self.realY:realY})
                g_loss, _ = sess.run([self.g_loss, opt_g], feed_dict={self.realX:realX, self.realY:realY})   
                dy_loss, _ = sess.run([self.dy_loss, opt_dy], feed_dict={self.realX:realX, self.realY:realY})                
                
                if l%50==0 and self.args.visualize:
                    t_realX, t_realY = sample_X_Y(256, 3, self.args.Xdir, self.args.Ydir, self.args.batch_size)
                    t_fakeX, t_fakeY = sess.run([self.fakeX, self.fakeY], feed_dict={self.realX:t_realX, self.realY:t_realY})
                    visualize(256, t_realX, t_fakeX, t_realY, t_fakeY, self.args.batch_size, l)

                if l%10000==0:
                    saver.save(sess, "save/model.ckpt")

                print(l,':','    g_loss:',g_loss,'    f_loss:',f_loss,'    dx_loss',dx_loss,'    dy_loss',dy_loss)
    
    def lrelu(self, x, leak=0.2, name="lrelu"):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def discriminator(self, x, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = tf.layers.conv2d(x, filters=64, kernel_size=[4,4], strides=(2,2), padding='SAME',name='d_h0_conv', reuse=reuse)
            h1 = tf.layers.batch_normalization(self.lrelu(tf.layers.conv2d(h0, filters=128, kernel_size=[4,4], strides=(2,2), padding='SAME', name='d_h1_conv', reuse=reuse)), name="d_bn_h1")
            h2 = tf.layers.batch_normalization(self.lrelu(tf.layers.conv2d(h1, filters=256, kernel_size=[4,4], strides=(2,2), padding='SAME', name='d_h2_conv', reuse=reuse)), name="d_bn_h2")
            h3 = tf.layers.batch_normalization(self.lrelu(tf.layers.conv2d(h2, filters=512, kernel_size=[4,4], strides=(1,1), padding='SAME', name='d_h3_conv', reuse=reuse)), name="d_bn_h3")
            out = tf.layers.conv2d(self.lrelu(h3), 1, kernel_size=[4,4], strides=(1,1), padding='SAME', name='d_out_conv', reuse=reuse)
            return out

    def genereter(self, x, name, reuse=False):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            e0 = tf.layers.batch_normalization(tf.layers.conv2d(x, filters=64, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_conv_enc0", reuse=reuse))
            e1 = tf.layers.batch_normalization(tf.layers.conv2d(self.lrelu(e0), filters=128, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_conv_e1", reuse=reuse))
            e2 = tf.layers.batch_normalization(tf.layers.conv2d(self.lrelu(e1), filters=256, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_conv_e2", reuse=reuse))
            e3 = tf.layers.batch_normalization(tf.layers.conv2d(self.lrelu(e2), filters=512, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_conv_e3", reuse=reuse))
            e4 = tf.layers.batch_normalization(tf.layers.conv2d(self.lrelu(e3), filters=512, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_conv_e4", reuse=reuse))
            e5 = tf.layers.batch_normalization(tf.layers.conv2d(self.lrelu(e4), filters=512, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_conv_e5", reuse=reuse))
            e6 = tf.layers.batch_normalization(tf.layers.conv2d(self.lrelu(e5), filters=512, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_conv_e6", reuse=reuse))
            e7 = tf.layers.batch_normalization(tf.layers.conv2d(self.lrelu(e6), filters=512, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_conv_e7", reuse=reuse))
            d0 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(e7), filters=512, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_dec_d0", reuse=reuse))
            d1 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(tf.concat([d0,e6], axis=3)), filters=512, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_dec_d1", reuse=reuse))
            d2 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(tf.concat([d1,e5], axis=3)), filters=512, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_dec_d2", reuse=reuse))
            d3 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(tf.concat([d2,e4], axis=3)), filters=512, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_dec_d3", reuse=reuse))
            d4 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(tf.concat([d3,e3], axis=3)), filters=256, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_dec_d4", reuse=reuse))
            d5 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(tf.concat([d4,e2], axis=3)), filters=128, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_dec_d5", reuse=reuse))
            d6 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(tf.concat([d5,e1], axis=3)), filters=64, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_dec_d6", reuse=reuse))
            d7 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(tf.concat([d6,e0], axis=3)), filters=3, kernel_size=[4,4], strides=(2,2), padding="SAME", name="g_dec_d7", reuse=reuse))
            return tf.nn.tanh(d7)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", dest="lr", type=float, default= 0.0002)
    parser.add_argument("--xdir", dest="Xdir", default="./Xdir/")
    parser.add_argument("--ydir", dest="Ydir", default="./Ydir/")
    parser.add_argument("--itrs", dest="itrs", type=int, default=3000000)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--visualize", dest="visualize", type=bool, default=True)
    parser.add_argument("--l1_lambda", dest="l1_lambda", type=float, default=50.0)
    parser.add_argument("--beta1", dest="beta1", type=float, default=0.5)
    args= parser.parse_args()
    
    if not os.path.exists('./save/'):
        os.mkdir('./save/')

    if not os.path.exists('./visualized/'):
        os.mkdir('./visualized/')

    model = model(args)
    model.train()

import tensorflow as tf
from Unet import UNet
from util import *

class Discriminator():
    def __init__(self, image, ini, reuse=False):
        with tf.variable_scope(ini) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h0 = self.lrelu(self.instance_norm(tf.layers.conv2d(image, filters=64, kernel_size=[3,3], strides=(2,2), padding='SAME',name='d_h0_conv'), "d_bn_h0"))
            h1 = self.lrelu(self.instance_norm(tf.layers.conv2d(h0, filters=128, kernel_size=[3,3], strides=(2,2), padding='SAME', name='d_h1_conv'), "d_bn_h1"))
            h2 = self.lrelu(self.instance_norm(tf.layers.conv2d(h1, filters=256, kernel_size=[3,3], strides=(2,2),padding='SAME', name='d_h2_conv'), "d_bn_h2"))
            h3 = self.lrelu(self.instance_norm(tf.layers.conv2d(h2, filters=512, kernel_size=[3,3], strides=(2,2),padding='SAME', name='d_h3_conv'), "d_bn_h3"))
            self.out = tf.layers.conv2d(h3, filters = 1, kernel_size=[3,3], strides=(1,1), name='d_out_conv')
            
    def lrelu(self, x, leak=0.2, name='lrelu'):
        return tf.maximum(x, leak * x)

    def instance_norm(self, input, name="_instance_norm"):
        with tf.variable_scope(name) as scope:
            depth = input.get_shape()[3]
            scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
            offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
            mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
            epsilon = 1e-5
            inv = tf.rsqrt(variance + epsilon)
            normalized = (input-mean)*inv
            return scale*normalized + offset

class Discriminator1():
    def __init__(self, x, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = tf.layers.conv2d(x, filters=64, kernel_size=[4,4], strides=(2,2), padding='SAME',name='d_h0_conv')
            h1 = tf.layers.batch_normalization(lrelu(tf.layers.conv2d(h0, filters=128, kernel_size=[4,4], strides=(2,2), padding='SAME', name='d_h1_conv')), name="d_bn_h1")
            h2 = tf.layers.batch_normalization(lrelu(tf.layers.conv2d(h1, filters=256, kernel_size=[4,4], strides=(2,2), padding='SAME', name='d_h2_conv')), name="d_bn_h2")
            h3 = tf.layers.batch_normalization(lrelu(tf.layers.conv2d(h2, filters=512, kernel_size=[4,4], strides=(1,1), padding='SAME', name='d_h3_conv')), name="d_bn_h3")
            out_ = tf.layers.conv2d(h3, filters = 1, kernel_size=[4,4], strides=(1,1), name='d_out_conv')
            self.out= tf.nn.sigmoid(out_)

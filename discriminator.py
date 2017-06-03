import tensorflow as tf
from Unet import UNet

class Discriminator():
    def __init__(self, image, ini, reuse=False):
        with tf.variable_scope(ini+"_discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h0 = self.lrelu(tf.layers.batch_normalization(tf.layers.conv2d(image, filters=64, kernel_size=[3,3], strides=(2,2), padding='SAME',name=ini+'_h0_conv')))
            h1 = self.lrelu(tf.layers.batch_normalization(tf.layers.conv2d(h0, filters=128, kernel_size=[3,3], strides=(2,2), padding='SAME', name=ini+'_h1_conv')))
            h2 = self.lrelu(tf.layers.batch_normalization(tf.layers.conv2d(h1, filters=256, kernel_size=[3,3], strides=(2,2),padding='SAME', name=ini+'_h2_conv')))
            h3 = self.lrelu(tf.layers.batch_normalization(tf.layers.conv2d(h2, filters=512, kernel_size=[3,3], strides=(2,2),padding='SAME', name=ini+'_h3_conv')))
            self.out = tf.layers.conv2d(h3, filters = 1, kernel_size=[3,3], strides=(1,1), name=ini+'_out_conv')
            
    def lrelu(self, x, leak=0.2, name='lrelu'):
        return tf.maximum(x, leak * x)

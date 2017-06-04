import tensorflow as tf
import numpy as np

class UNet():
    def __init__(self, inputs, ini, reuse):
        self.ini = ini
        self.reuse = reuse
        with tf.variable_scope(ini+"_genereter"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False;
            
            enc_c0 = tf.nn.relu(self.instance_norm(tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_enc_c0'), "g_bn_c0"))
            enc_c1 = tf.nn.relu(self.instance_norm(tf.layers.conv2d(inputs=enc_c0, filters=64, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_enc_c1'), "g_bn_c1"))
            enc_c2 = tf.nn.relu(self.instance_norm(tf.layers.conv2d(inputs=enc_c1, filters=64, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_enc_c2'), "g_bn_c2"))
            enc_c3 = tf.nn.relu(self.instance_norm(tf.layers.conv2d(inputs=enc_c2, filters=128, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_enc_c3'), "g_bn_c3"))
            enc_c4 = tf.nn.relu(self.instance_norm(tf.layers.conv2d(inputs=enc_c3, filters=128, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_enc_c4'), "g_bn_c4"))
            enc_c5 = tf.nn.relu(self.instance_norm(tf.layers.conv2d(inputs=enc_c4, filters=256, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_enc_c5'), "g_bn_c5"))
            enc_c6 = tf.nn.relu(self.instance_norm(tf.layers.conv2d(inputs=enc_c5, filters=256, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_enc_c6'), "g_bn_c6"))
            enc_c7 = tf.nn.relu(self.instance_norm(tf.layers.conv2d(inputs=enc_c6, filters=512, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_enc_c7'), "g_bn_c7"))
            enc_c8 = tf.nn.relu(self.instance_norm(tf.layers.conv2d(inputs=enc_c7, filters=512, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_enc_c8'), "g_bn_c8"))

            dec_dc8 = tf.nn.relu(tf.layers.batch_normalization((tf.layers.conv2d_transpose(tf.concat([enc_c7,enc_c8],3), filters=512, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_dec_dc8'), "g_bn_dc8"))
            dec_dc7 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dec_dc8, filters=256, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_dec_dc7'), "g_bn_dc7"))
            dec_dc6 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.concat([enc_c6,dec_dc7],3), filters=256, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_dec_dc6'), "g_bn_dc6"))
            dec_dc5 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dec_dc6, filters=128, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_dec_dc5'), "g_bn_dc5"))
            dec_dc4 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.concat([enc_c4,dec_dc5],3), filters=128, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_dec_dc4'), "g_bn_dc4"))
            dec_dc3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dec_dc4, filters=64, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_dec_dc3'), "g_bn_dc3"))
            dec_dc2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.concat([enc_c2,dec_dc3],3), filters=64, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_dec_dc2'), "g_bn_dc2"))
            dec_dc1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dec_dc2, filters=32, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_dec_dc1'), "g_bn_dc1"))
            self.dec_dc0 = tf.layers.conv2d(tf.concat([enc_c0,dec_dc1],3), filters=3, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_dec_dc0')

    def instance_norm(self, input, name="instance_norm"):
        with tf.variable_scope(name):
            depth = input.get_shape()[3]
            scale = tf.get_variable(self.ini+"scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
            offset = tf.get_variable(self.ini+"offset", [depth], initializer=tf.constant_initializer(0.0))
            mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
            epsilon = 1e-5
            inv = tf.rsqrt(variance + epsilon)
            normalized = (input-mean)*inv
            return scale*normalized + offset

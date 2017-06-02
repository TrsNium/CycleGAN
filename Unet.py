import tensorflow as tf
import numpy as np

class UNet():
    def __init__(self, inputs, ini, reuse):
        with tf.variable_scope(ini+"genereter"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False;

            enc_c0 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_enc_c0'),name=ini+'_bn_c0'))
            enc_c1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c0, filters=64, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_enc_c1'),name=ini+'_bn_c1'))
            enc_c2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c1, filters=64, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_enc_c2'),name=ini+'_bn_c2'))
            enc_c3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c2, filters=128, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_enc_c3'),name=ini+'_bn_c3'))
            enc_c4 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c3, filters=128, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_enc_c4'),name=ini+'_bn_c4'))
            enc_c5 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c4, filters=256, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_enc_c5'),name=ini+'_bn_c5'))
            enc_c6 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c5, filters=256, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_enc_c6'),name=ini+'_bn_c6'))
            enc_c7 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c6, filters=512, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_enc_c7'),name=ini+'_bn_c7'))
            enc_c8 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c7, filters=512, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_enc_c8'),name=ini+'_bn_c8'))

            dec_dc8 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.concat([enc_c7,enc_c8],3), filters=512, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_dec_dc8'),name=ini+'_bn_d8'))
            dec_dc7 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dec_dc8, filters=256, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_dec_dc7'),name=ini+'_bn_d7'))
            dec_dc6 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.concat([enc_c6,dec_dc7],3), filters=256, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_dec_dc6'),name=ini+'_bn_d6'))
            dec_dc5 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dec_dc6, filters=128, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_dec_dc5'),name=ini+'_bn_d5'))
            dec_dc4 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.concat([enc_c4,dec_dc5],3), filters=128, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_dec_dc4'),name=ini+'_bn_d4'))
            dec_dc3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dec_dc4, filters=64, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_dec_dc3'),name=ini+'_bn_d3'))
            dec_dc2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.concat([enc_c2,dec_dc3],3), filters=64, kernel_size=[4,4], strides=(2,2), padding='SAME', name=ini+'_dec_dc2'),name=ini+'_bn_d2'))
            dec_dc1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dec_dc2, filters=32, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_dec_dc1'),name=ini+'_bn_d1'))
            self.dec_dc0 = tf.layers.conv2d(tf.concat([enc_c0,dec_dc1],3), filters=3, kernel_size=[3,3], strides=(1,1), padding='SAME', name=ini+'_dec_dc0')

# -*- coding: utf-8 -*-
import tensorflow as tf

DATA_FORMAT = "channels_first"


class ResNet(object):
    def __init__(self, board_size):
        self.board_size = board_size
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2, board_size, board_size])
        self.value = None
        self.policy = None
        self.network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def network(self):
        f = self.inputs
        with tf.variable_scope("bone"):
            f = tf.layers.conv2d(f, 32, 3, padding="SAME", data_format=DATA_FORMAT, name="conv1", activation=tf.nn.elu)
            f = tf.layers.conv2d(f, 64, 3, padding="SAME", data_format=DATA_FORMAT, name="conv2", activation=tf.nn.elu)
            f = tf.layers.conv2d(f, 128, 3, padding="SAME", data_format=DATA_FORMAT, name="conv3", activation=tf.nn.elu)

        with tf.variable_scope("value"):
            v = tf.layers.conv2d(f, 32, 1, padding="VALID", data_format=DATA_FORMAT, name="conv1", activation=tf.nn.elu)
            v = tf.reshape(v, (tf.shape(v)[0], -1))
            v = tf.layers.dense(v, 256, activation=tf.nn.elu, name="fc1")
            v = tf.layers.dense(v, 64, activation=tf.nn.elu, name="fc2")
            self.value = tf.layers.dense(v, 1, activation=tf.nn.tanh, name="fc3")

        with tf.variable_scope("policy"):
            p = tf.layers.conv2d(f, 32, 1, padding="VALID", data_format=DATA_FORMAT, name="conv1", activation=tf.nn.elu)
            p = tf.reshape(p, (tf.shape(p)[0], -1))
            self.policy = tf.layers.dense(p, self.board_size*self.board_size, activation=None, name="fc1")

    def eval(self, inputs):
        prob = tf.nn.softmax(self.policy, axis=1)
        prob_, value_ = self.sess.run([prob, self.value], feed_dict={self.inputs: inputs})
        return prob_, value_

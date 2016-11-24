import argparse
# Import data
import tensorflow as tf
from data_util import DataUtil
from config import Config


class Coref_clustter:
    def __init__(self):
        self.config = Config()
        self.du = DataUtil(self.config)
        self.W1 = tf.Variable(tf.zeros([self.config.M1, self.config.I]))
        self.b1 = tf.Variable(tf.zeros([self.config.M1, 1]))
        self.W2 = tf.Variable(tf.zeros([self.config.M2, self.config.M1]))
        self.b2 = tf.Variable(tf.zeros([self.config.M2, 1]))
        self.W3 = tf.Variable(tf.zeros([self.config.D, self.config.M2]))
        self.b3 = tf.Variable(tf.zeros([self.config.D, 1]))
        self.Wm = tf.Variable(tf.zeros([1, self.config.D]))
        self.bm = tf.Variable(0)
        self.M = tf.placeholder([])
        self.As = tf.placeholder([])
        self.Ts = tf.placeholder([])

    def h(self,a,m):
        return tf.constant(tf.random_normal([self.config.I, 1]))

    def r(self,a,m):
        h1 = tf.nn.relu(tf.matmul(self.W1, self.h(a, m)) + self.b1)
        h2 = tf.nn.relu(tf.matmul(self.W2, h1) + self.b2)
        y = tf.nn.relu(tf.matmul(self.W3, h2) + self.b3)
        return y

    def s(self,a,m):
        y = self.r(a,m)
        s_val = tf.matmul(self.Wm, y) + self.bm
        return s_val

    def mistake(self,a, T):
        if a==self.config.NA and T[0]!=[self.config.NA]:
            return self.config.a_fn
        if a!=self.config.NA and T[0]==[self.config.NA]:
            return self.config.a_fa
        if a!=self.config.NA and a not in T:
            return self.config.a_wl
        return 0

    def read_feed(self):
        return {}


    def main(self):
        loss = 0
        for i in range(len(self.M)):
            m = self.M[i]
            A = self.As[i]
            loss = 0
            T = self.Ts[i]
            max_subloss = 0
            for a in A:
                max_st = tf.reduce_max([self.s(t,m) for t in T])
                sm = self.s(a,m)
                mis = self.mistake(a, T)
                subloss = mis * (1 + sm - max_st)
                if loss > max_subloss:
                    max_subloss = subloss
            loss += max_subloss

        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        sess = tf.InteractiveSession()
        # Train
        tf.initialize_all_variables().run()
        for _ in range(1000):
            feed = self.du.read_feed()
            sess.run(train_step, feed_dict=feed)

            # Test trained model
        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(sess.run(accuracy, feed_dict={As: mnist.test.images,
        #                                     Ts: mnist.test.labels,
        #                                     M: []}))

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./MNIST_data',
                        help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()

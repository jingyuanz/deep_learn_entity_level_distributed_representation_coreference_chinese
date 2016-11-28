#coding=utf-8
import argparse
# Import data
import tensorflow as tf
from compiler.ast import flatten
from data_util import DataUtil
from config import Config
import polyglot
from polyglot.text import Text
from polyglot.mapping import Embedding
from tensorflow import TensorShape
import numpy as np
from numpy import ndarray as nd


class Coref_clustter:
    def __init__(self):
        self.config = Config()
        self.du = DataUtil(self.config)
        self.embeddings = self.du.embeddings
        self.W1 = tf.Variable(tf.zeros([self.config.M1, self.config.I]))
        self.b1 = tf.Variable(tf.zeros([self.config.M1, 1]))
        self.W2 = tf.Variable(tf.zeros([self.config.M2, self.config.M1]))
        self.b2 = tf.Variable(tf.zeros([self.config.M2, 1]))
        self.W3 = tf.Variable(tf.zeros([self.config.D, self.config.M2]))
        self.b3 = tf.Variable(tf.zeros([self.config.D, 1]))
        self.Wm = tf.Variable(tf.zeros([1, self.config.D]))
        self.bm = tf.Variable(0.0)
        # self.M = tf.placeholder(shape=TensorShape([]),dtype=tf.float32)
        # self.As = tf.placeholder(shape=TensorShape([]),dtype=tf.float32)
        # self.Ts = tf.placeholder(shape=TensorShape([]),dtype=tf.float32)
        self.M = []
        self.As = []
        self.Ts = []

    def h(self, a, m):
        if a=='#':
            a = m
        print a,m
        embed_a = nd.tolist(self.embeddings[a[2]])
        # print len(embed_a)
        embed_m = nd.tolist(self.embeddings[m[2]])
        # print len(embed_m)
        first_aw_embed = nd.tolist(self.du.find_first_word_embedding(a))
        # print len(first_aw_embed)
        first_mw_embed = nd.tolist(self.du.find_first_word_embedding(m))
        # print len(first_mw_embed)
        last_aw_embed = nd.tolist(self.du.find_last_word_embedding(a))
        # print len(last_aw_embed)
        last_mw_embed = nd.tolist(self.du.find_last_word_embedding(m))
        # print len(last_mw_embed)
        proced2_a_embed = flatten([nd.tolist(self.embeddings[word]) for word in self.du.find_proceding(a, 2) if word != ''])
        proced2_a_embed = proced2_a_embed + [0.0]*(512-len(proced2_a_embed))
        # print len(proced2_a_embed)
        follow2_a_embed = flatten([nd.tolist(self.embeddings[word]) for word in self.du.find_following(a, 2) if word != ''])
        follow2_a_embed = follow2_a_embed + [0.0]*(512-len(follow2_a_embed))
        # print len(follow2_a_embed)

        proced2_m_embed = flatten([nd.tolist(self.embeddings[word]) for word in self.du.find_proceding(m, 2) if word != ''])
        proced2_m_embed = proced2_m_embed + [0.0] * (512 - len(proced2_m_embed))
        # print len(proced2_m_embed)
        follow2_m_embed = flatten([nd.tolist(self.embeddings[word]) for word in self.du.find_following(m, 2) if word != ''])
        follow2_m_embed = follow2_m_embed + [0.0] * (512 - len(follow2_m_embed))
        # print len(first_mw_embed)

        avg5f_a = self.du.calc_word_average(self.du.find_following(a, 5))
        # print len(avg5f_a)
        avg5p_a = self.du.calc_word_average(self.du.find_proceding(a, 5))
        # print len(avg5p_a)
        avg5f_m = self.du.calc_word_average(self.du.find_following(m, 5))
        # print len(avg5f_m)
        avg5p_m = self.du.calc_word_average(self.du.find_proceding(m, 5))
        # print len(avg5p_m)
        avgsent_a = self.du.average_sent(a)
        # print len(avgsent_a)
        avgsent_m = self.du.average_sent(m)
        # print len(avgsent_m)
        avg_all = [self.du.all_word_average]
        # print len(avg_all)
        type_a = [0]  # self.du.type_dict[a[3]]
        type_m = [0]  # self.du.type_dict[m[3]]
        mention_pos_a = self.du.mention_pos(a)
        mention_pos_m = self.du.mention_pos(m)

        mention_len_a = [len(a[2])]
        mention_len_m = [len(m[2])]

        distance = self.du.distance_mentions(a, m)
        distance_m = self.du.distance_intervening_mentions(a, m)

        result = embed_a + first_aw_embed + last_aw_embed + proced2_a_embed + follow2_a_embed + avg5f_a + avg5p_a + avgsent_a + type_a + mention_pos_a + mention_len_a + embed_m + first_mw_embed + last_mw_embed + proced2_m_embed + follow2_m_embed + avg5f_m + avg5p_m + avgsent_m + type_m + mention_pos_m + mention_len_m + avg_all + distance + distance_m
        # print len(result) #4873
        matrix_result = tf.expand_dims(tf.constant(result), 1)
        # print matrix_result
        return matrix_result

    def r(self, a, m):
        h1 = tf.nn.relu(tf.matmul(self.W1, self.h(a, m)) + self.b1)
        h2 = tf.nn.relu(tf.matmul(self.W2, h1) + self.b2)
        y = tf.nn.relu(tf.matmul(self.W3, h2) + self.b3)
        return y

    def s(self, a, m):
        y = self.r(a, m)
        s_val = tf.matmul(self.Wm, y) + self.bm
        return s_val

    def mistake(self, a, T):
        if a == self.config.NA and T[0] != [self.config.NA]:
            return self.config.a_fn
        if a != self.config.NA and T[0] == [self.config.NA]:
            return self.config.a_fa
        if a != self.config.NA and a not in T:
            return self.config.a_wl
        return 0

    def main(self):
        # loss = tf.reduce_sum()
        # print typeof(self.M)
        for i in range(len(self.M)):
            m = self.M[i]
            A = self.As[i]
            loss = 0
            T = self.Ts[i]
            max_subloss = 0
            for a in A:
                max_st = tf.reduce_max([self.s(t, m) for t in T])
                sm = self.s(a, m)
                mis = self.mistake(a, T)
                subloss = mis * (1 + sm - max_st) * 1.0
                if loss > max_subloss:
                    max_subloss = subloss
            loss += max_subloss
        loss = np.float32(loss)
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        sess = tf.InteractiveSession()
        # Train
        tf.initialize_all_variables().run()
        for i in range(20):
            feed = self.du.build_feed_dict(self.config.batch_size * i, self.config.batch_size * (i + 1))

            sess.run(train_step, feed_dict=feed)

            # Test trained model
            # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # print(sess.run(accuracy, feed_dict={As: mnist.test.images,
            #                                     Ts: mnist.test.labels,
            #                                     M: []}))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


if __name__ == '__main__':

    cc = Coref_clustter()
    # cc.h(cc.M[0],cc.M[1])
    cc.main()

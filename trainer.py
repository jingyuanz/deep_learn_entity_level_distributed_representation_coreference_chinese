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
import sys
import random


class Coref_clustter:
    def __init__(self):
        self.config = Config()
        self.du = DataUtil(self.config)
        self.embeddings = self.du.embeddings
        self.W1 = tf.get_variable("w1",shape=[self.config.M1, self.config.I])
        self.b1 = tf.get_variable("b1",shape=[self.config.M1, 1])
        self.W2 = tf.get_variable("w2",shape=[self.config.M2, self.config.M1])
        self.b2 = tf.get_variable("b2",shape=[self.config.M2, 1])
        self.W3 = tf.get_variable("w3",shape=[self.config.D, self.config.M2])
        self.b3 = tf.get_variable("b3",shape=[self.config.D, 1])
        self.Wm = tf.get_variable("wm",shape=[1, self.config.D])
        self.bm = tf.get_variable("bm",shape=[1])
        # self.M = tf.placeholder(shape=TensorShape([]),dtype=tf.float32)
        # self.As = tf.placeholder(shape=TensorShape([]),dtype=tf.float32)
        # self.Ts = tf.placeholder(shape=TensorShape([]),dtype=tf.float32)
        self.M = self.du.mentions
        self.As = self.du.As
        self.Ts = self.du.Ts
        self.du.max_as_count += 1
        self.mistakes = tf.placeholder(tf.float32, shape=[self.config.batch_size, self.du.max_as_count])
        self.batch_HAs = tf.placeholder(tf.float32,shape=[self.config.batch_size, self.du.max_as_count, self.config.I])
        self.batch_hts = tf.placeholder(tf.float32,shape=[self.config.batch_size, self.config.I])
        self.indices = tf.placeholder(tf.float32, shape=[self.config.batch_size])
        self.test_h_r_antecedents = tf.placeholder(tf.float32, shape=[self.config.test_batch_size, self.du.max_as_count, self.config.I])
        # self.test_h_r_answers = tf.placeholder(tf.float32, shape=[self.config.test_batch_size, self.config.I])
        self.test_indices = tf.placeholder(tf.int64, shape=[self.config.test_batch_size])
        self.test_answers_indices = tf.placeholder(tf.int64, shape=[self.config.test_batch_size])

    def h(self, a, m):
        if a == 0 and m == 0:
            result = [np.float32(0.0)]*self.config.I
            return result
        if a=='#':
            a = m
        embed_a = nd.tolist(self.embeddings.get(a[2],a[3],default=np.asarray([0.0]*self.config.embedding_size)))
        embed_m = nd.tolist(self.embeddings.get(m[2],m[3],default=np.asarray([0.0]*self.config.embedding_size)))
        # print len(embed_m)
        first_aw_embed = nd.tolist(self.du.find_first_word_embedding(a))
        # print len(first_aw_embed)
        first_mw_embed = nd.tolist(self.du.find_first_word_embedding(m))
        # print len(first_mw_embed)
        last_aw_embed = nd.tolist(self.du.find_last_word_embedding(a))
        # print len(last_aw_embed)
        last_mw_embed = nd.tolist(self.du.find_last_word_embedding(m))
        # print len(last_mw_embed)
        proced2_a_embed = self.du.find_proceding_embeddings(a, 2)
        follow2_a_embed = self.du.find_following_embeddings(a, 2)

        proced2_m_embed = self.du.find_proceding_embeddings(m, 2)
        follow2_m_embed = self.du.find_following_embeddings(m, 2)

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
        type_a = [self.du.t_dict[a[3]]]  # self.du.type_dict[a[3]]
        type_m = [self.du.t_dict[m[3]]]  # self.du.type_dict[m[3]]
        mention_pos_a = self.du.mention_pos(a)
        mention_pos_m = self.du.mention_pos(m)

        mention_len_a = [len(a[2])]
        mention_len_m = [len(m[2])]

        distance = self.du.distance_mentions(a, m)
        distance_m = self.du.distance_intervening_mentions(a, m)

        result = embed_a + first_aw_embed + last_aw_embed + proced2_a_embed + follow2_a_embed + avg5f_a + avg5p_a + avgsent_a + type_a + mention_pos_a + mention_len_a + embed_m + first_mw_embed + last_mw_embed + proced2_m_embed + follow2_m_embed + avg5f_m + avg5p_m + avgsent_m + type_m + mention_pos_m + mention_len_m + avg_all + distance + distance_m
        if len(result)!=self.config.I:
            print len(proced2_a_embed)
            print len(follow2_a_embed)
            print len(proced2_m_embed)
            print len(follow2_m_embed)

            print len(result) #4873
            print
            sys.exit(0)
        # print matrix_result
        # if len(result)!=self.config.embedding_size:
        #     print len(result)
        return result

    def r(self, h):
        h1 = tf.nn.relu(tf.matmul(self.W1,tf.reshape(h,[self.config.I, 1])) + self.b1)
        h2 = tf.nn.relu(tf.matmul(self.W2,h1) + self.b2)
        y = tf.nn.relu(tf.matmul(self.W3,h2) + self.b3)
        return y

    def s(self, h):
        y = self.r(h)
        s_val = tf.matmul(self.Wm, y) + self.bm
        # s_val = tf.sigmoid(s_val)
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
        ''' up to here'''
        self.loss = tf.reduce_sum(tf.map_fn(lambda index: tf.reduce_max(self.mistakes[tf.to_int32(index)]*tf.map_fn(lambda x: 1+self.s(x)-self.s(self.batch_hts[tf.to_int32(index)]), self.batch_HAs[tf.to_int32(index)])), self.indices))
        train_step = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(self.loss)
        # prediction = tf.map_fn(lambda index: self.test_h_r_antecedents[tf.to_int32(tf.arg_max(tf.map_fn(lambda h: self.s(h), self.test_h_r_antecedents[tf.to_int32(index)]), 1))], self.test_indices)
        self.prediction = tf.map_fn(lambda index: tf.arg_max(tf.map_fn(lambda h: self.s(h), self.test_h_r_antecedents[tf.to_int32(index)]),0), self.test_indices)
        self.prediction = tf.squeeze(self.prediction)
        self.correct_prediction = tf.equal(self.prediction, self.test_answers_indices)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        sess = tf.InteractiveSession()
        # Train
        tf.initialize_all_variables().run()
        for i in range(100):
            print "epoch:", i
            # feed = self.du.build_feed_dict(self.config.batch_size * i, self.config.batch_size * (i + 1))
            # loss = tf.reduce_sum(tf.map_fn(lambda index: tf.reduce_max(tf.squeeze(self.mistakes)[index] * tf.map_fn(
            #     lambda x: 1 + self.s(x) - self.s(tf.squeeze(self.hts)[index]), tf.squeeze(self.HAs)[index])),
            #                                self.indices))
            # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
            batch_Ms = self.M[i*self.config.batch_size:(i+1)*self.config.batch_size]
            batch_As = self.As[i*self.config.batch_size:(i+1)*self.config.batch_size]
            batch_Ts = self.Ts[i*self.config.batch_size:(i+1)*self.config.batch_size]
            mistakes = []
            for k in range(len(batch_Ts)):
                T = batch_Ts[k]
                A = batch_As[k]
                mistake = [np.float32(self.mistake(a, T)) for a in A]
                mistake.extend([np.float32(0.0)] * (self.du.max_as_count - len(mistake)))
                mistakes.append(mistake)
            # print "mistakes:",mistakes, len(mistakes).
            reduced_Ts = [T[0] for T in batch_Ts]
            # for mt_i in range(len(batch_Ms)):
            #     T_w = reduced_Ts[mt_i]
            #     M_w = batch_Ms[mt_i]
            #     if T_w == '#':
            #         print T_w,
            #     else:
            #         print T_w[2],
            #     print M_w[2],
            #     print "///",
            # print

            batch_hts = []
            for j in range(len(reduced_Ts)):
                t = reduced_Ts[j]
                m = batch_Ms[j]
                ht = self.h(t, m)
                batch_hts.append(ht)
            # hts = np.array(hts)
            batch_HAs = []
            for z in range(len(batch_Ms)):
                As = batch_As[z]
                m = batch_Ms[z]
                HA = [self.h(a,m) for a in As]
                padding = [np.float32(0.0)]*self.config.I
                HA.extend([padding]*(self.du.max_as_count-len(HA)))
                batch_HAs.append(HA)
            # batch_HAs = tf.convert_to_tensor(batch_HAs)
            # HAs = np.array(HAs)
            # print "HAs: ",HAs
            indices = [w for w in range(self.config.batch_size)]
            assert len(batch_HAs) == len(batch_hts) == len(mistakes)

            test_r_answers, test_r_antecedents = self.du.get_test_data(self.config.test_batch_size)
            # for r_a_i in range(len(test_r_answers)):
            #     test_r = test_r_answers[r_a_i][1]
            #     test_a = test_r_answers[r_a_i][0]
            #     print test_r[2],
            #     print test_a[2]
            #     test_ands = test_r_antecedents[r_a_i]
            #     for test_an in test_ands:
            #         test_an_an = test_an[0]
            #         if test_an_an == '#':
            #             print test_an_an,
            #         else:
            #             print test_an_an[2],
            #     print
            #     print

            test_h_r_answers = test_r_answers
            test_h_r_antecedents = [map(lambda x: self.h(x[0],x[1]), test_r_antecedents_batch) for test_r_antecedents_batch in test_r_antecedents]
            test_indices = [ti for ti in range(self.config.test_batch_size)]
            assert len(test_h_r_answers) == len(test_h_r_antecedents) == len(test_indices)

            # train_accuracy = self.accuracy.eval(feed_dict={
            #     self.test_h_r_answers: test_h_r_answers, self.test_h_r_antecedents: test_h_r_antecedents, self.test_indices: test_indices})
            a,b,c,e,f = sess.run([self.loss,train_step, self.prediction, self.correct_prediction, self.accuracy], feed_dict={self.mistakes: mistakes, self.batch_hts: batch_hts, self.batch_HAs: batch_HAs, self.indices: indices,self.test_answers_indices: test_h_r_answers, self.test_h_r_antecedents: test_h_r_antecedents, self.test_indices: test_indices})
            # sess.run(train_step,feed_dict={self.mistakes: mistakes, self.batch_hts: batch_hts, self.batch_HAs: batch_HAs, self.indices: indices})
            print 'prediction_maxs: \n',
            for p_i in range(len(c)):
                ans = test_r_antecedents[p_i][c[p_i]][0]
                if ans!='#':
                    ans = ans[2]
                print 'predict: ', c[p_i], ans, 'labelled: ', test_h_r_answers[p_i], test_r_antecedents[p_i][c[p_i]][0][2]
            print 'correct/incorrect: \n', e
            print 'accuracy: \n', f
            print 'loss: \n',a
            print



            # Test trained model
            # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # print(sess.run(accuracy, feed_dict={As: mnist.test.images,
            #                                     Ts: mnist.test.labels,
            #                                     M: []  }))

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

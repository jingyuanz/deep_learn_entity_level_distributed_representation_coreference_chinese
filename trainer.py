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
        # self.R = self.du.test_rs
        # self.As = self.du.test_r_antecedents
        # self.Ts = self.du.test_r_answers
        self.du.max_as_count += 1
        self.mistakes = tf.placeholder(tf.float32, shape=[self.config.batch_size, self.du.max_as_count])
        self.batch_HAs = tf.placeholder(tf.float32,shape=[self.config.batch_size, self.du.max_as_count, self.config.I])
        self.batch_hts = tf.placeholder(tf.float32,shape=[self.config.batch_size, self.config.I])
        self.indices = tf.placeholder(tf.float32, shape=[self.config.batch_size])
        self.test_h_r_antecedents = tf.placeholder(tf.float32, shape=[self.config.test_batch_size, self.du.max_as_count, self.config.I])
        # self.test_h_r_answers = tf.placeholder(tf.float32, shape=[self.config.test_batch_size, self.config.I])
        self.test_indices = tf.placeholder(tf.float32, shape=[self.config.test_batch_size, self.du.max_as_count])
        self.test_indices2 = tf.placeholder(tf.int64, shape=[self.config.test_batch_size])
        self.test_answers_indices = tf.placeholder(tf.int64, shape=[self.config.test_batch_size])

        self.train_h_r_antecedents = tf.placeholder(tf.float32, shape=[self.config.test_batch_size, self.du.max_as_count,
                                                                      self.config.I])
        # self.test_h_r_answers = tf.placeholder(tf.float32, shape=[self.config.test_batch_size, self.config.I])
        self.train_indices = tf.placeholder(tf.float32, shape=[self.config.test_batch_size, self.du.max_as_count])
        self.train_indices2 = tf.placeholder(tf.int64, shape=[self.config.test_batch_size])
        self.train_answers_indices = tf.placeholder(tf.int64, shape=[self.config.test_batch_size])


    def r(self, h):
        h1 = tf.nn.relu(tf.matmul(self.W1,tf.reshape(h,[self.config.I, 1])) + self.b1)
        h2 = tf.nn.relu(tf.matmul(self.W2,h1) + self.b2)
        y = tf.nn.relu(tf.matmul(self.W3,h2) + self.b3)
        return y

    def s(self, h):
        y = self.r(h)
        s_val = tf.matmul(self.Wm, y) + self.bm
        # s_val = tf.sigmoid(s_val)
        return abs(s_val/10.0)



    def main(self):
        ''' up to here'''
        # self.temp2 = tf.map_fn(lambda index: self.mistakes[tf.to_int32(index)]
        self.temp1 = tf.map_fn(lambda index: tf.reduce_max(self.mistakes[tf.to_int32(index)]*tf.squeeze(tf.map_fn(lambda x: 1+self.s(x)-self.s(self.batch_hts[tf.to_int32(index)]), self.batch_HAs[tf.to_int32(index)]))), self.indices)
        # self.temp1 = tf.map_fn(lambda index: tf.reduce_mean((self.mistakes[tf.to_int32(index)]*tf.squeeze(tf.map_fn(lambda x: 5+self.s(x)-self.s(self.batch_hts[tf.to_int32(index)]), self.batch_HAs[tf.to_int32(index)]))),, self.indices)

        self.loss = tf.reduce_sum(self.temp1)
        train_step = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(self.loss)
        # prediction = tf.map_fn(lambda index: self.test_h_r_antecedents[tf.to_int32(tf.arg_max(tf.map_fn(lambda h: self.s(h), self.test_h_r_antecedents[tf.to_int32(index)]), 1))], self.test_indices)
        '''for testing'''
        self.prediction = tf.squeeze(tf.map_fn(lambda index: tf.map_fn(lambda h: self.s(h), self.test_h_r_antecedents[tf.to_int32(index[0])]), self.test_indices))
        self.prediction2 = tf.map_fn(lambda index: tf.arg_max(
            tf.squeeze(tf.map_fn(lambda h: self.s(h), self.test_h_r_antecedents[tf.to_int32(index)])), 0),
                                     self.test_indices2)
        self.prediction2 = tf.squeeze(self.prediction2)

        self.correct_prediction = tf.equal(self.prediction2, self.test_answers_indices)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        '''for training'''
        self.prediction_train = tf.squeeze(
            tf.map_fn(lambda index: tf.map_fn(lambda h: self.s(h), self.train_h_r_antecedents[tf.to_int32(index[0])]),
                      self.train_indices))
        self.prediction2_train = tf.map_fn(lambda index: tf.arg_max(
            tf.squeeze(tf.map_fn(lambda h: self.s(h), self.train_h_r_antecedents[tf.to_int32(index)])), 0),
                                     self.train_indices2)
        self.prediction2_train = tf.squeeze(self.prediction2_train)

        self.correct_prediction_train = tf.equal(self.prediction2_train, self.train_answers_indices)
        self.accuracy_train = tf.reduce_mean(tf.cast(self.correct_prediction_train, tf.float32))


        sess = tf.InteractiveSession()
        # Train
        tf.initialize_all_variables().run()

        for i in range(100):
            print "epoch:", i

            shuffled_epoch_Rs, shuffled_epoch_HAs, shuffled_epoch_HTs, shuffled_epoch_mistakes, shuffled_answer_indices = self.du.get_shuffled_data_set()
            print 'epoch data fetched'
            start_i = 0
            len_data_set = len(shuffled_epoch_Rs)
            step = 1

            while start_i < len_data_set:

                print "epoch",i," step", step
                step += 1
                end_i = start_i + self.config.batch_size
                if end_i > len_data_set:
                    end_i = len_data_set
                    start_i = end_i - self.config.batch_size
                batch_Rs = shuffled_epoch_Rs[start_i:end_i]
                batch_As = shuffled_epoch_HAs[start_i:end_i]
                batch_Ts = shuffled_epoch_HTs[start_i:end_i]
                batch_mistakes = shuffled_epoch_mistakes[start_i:end_i]
                print 'step data fetched'
                batch_HAs, batch_HTs = self.du.encode_mention_pairs(batch_Rs, batch_Ts, batch_As)
                print 'step data encoded'
                indices = [w for w in range(self.config.batch_size)]

                start_i = end_i
                print 'training'
                _, batch_loss, _ = sess.run([self.temp1,self.loss,train_step], feed_dict={self.mistakes: batch_mistakes, self.batch_hts: batch_HTs, self.batch_HAs: batch_HAs, self.indices: indices})

            print 'epoch training finished'
            print 'training....testing...'
            test_rs_batch, test_answer_indices, test_r_antecedents = self.du.get_test_data(self.config.test_batch_size, 'test')
            train_rs_batch, train_answer_indices, train_r_antecedents = self.du.get_test_data(self.config.test_batch_size, 'train')

            test_indices = [[ti for tii in range(self.du.max_as_count)] for ti in range(self.config.test_batch_size)]
            test_indices2 = [ti2 for ti2 in range(self.config.test_batch_size)]
            train_indices = [[ti for tii in range(self.du.max_as_count)] for ti in range(self.config.test_batch_size)]
            train_indices2 = [ti2 for ti2 in range(self.config.test_batch_size)]

            test_predict_by_scores, test_predict_indices, test_true_false, test_accuracy, train_predict_by_scores, train_predict_indices, train_true_false, train_accuracy = sess.run([self.prediction, self.prediction2, self.correct_prediction, self.accuracy, self.prediction_train, self.prediction2_train, self.correct_prediction_train, self.accuracy_train], feed_dict={self.test_answers_indices: test_answer_indices, self.test_h_r_antecedents: test_r_antecedents, self.test_indices: test_indices, self.test_indices2: test_indices2,self.train_answers_indices: train_answer_indices, self.train_h_r_antecedents: train_r_antecedents, self.train_indices: train_indices, self.train_indices2: train_indices2})


            '''print stuff'''

            print
            # print 'predictions: \n',nd.tolist(c)
            print 'prediction indices: \n', nd.tolist(test_predict_indices)
            print 'actual predicts: \n',test_answer_indices
            for p_i in range(len(test_predict_indices)):
                ans = test_r_antecedents[p_i][test_predict_indices[p_i]]
                if ans!=self.config.NA:
                    ans = ans[2]
                label = test_r_antecedents[p_i][test_answer_indices[p_i]]
                sent_num = self.config.NA
                w_num = self.config.NA
                if label != self.config.NA:
                    w_num = label[1]
                    sent_num = label[0]
                    label = label[2]

                print 'predict: ', test_predict_indices[p_i], ans, 'labelled: ', test_answer_indices[p_i], label, w_num, sent_num
            print 'correct/incorrect: \n', test_true_false
            print 'test_accuracy: \n', test_accuracy
            print 'train_accuracy: \n', train_accuracy

            print


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

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 19:40:51 2016

@author: sean
"""
from data_util import DataUtil
from config import Config
from sklearn import metrics
import numpy as np
import tensorflow as tf
import random, time, os


class Coref_cluster(object):

    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholder()
        scores = self.add_model()
        self.add_loss_and_train_op(scores)
        self.add_predict_op(scores)
        self.init_op = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

    def load_data(self):
        self.du = DataUtil(self.config)
        self.max_as_count = self.du.max_as_count

    def add_placeholder(self):
        self.inputs = tf.placeholder(tf.float32)
        self.labels = tf.placeholder(tf.int32)
        self.deltas = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs, deltas=None, labels=None):
        feed = {self.inputs: inputs}
        if labels:
            feed[self.deltas] = deltas
            feed[self.labels] = labels
        return feed

    def add_model(self):
        x = tf.reshape(self.inputs, (-1, self.config.I))
        W1 = tf.get_variable('W1', [self.config.I, self.config.M1])
        b1 = tf.get_variable('b1', [self.config.M1])
        fc1 = tf.matmul(x, W1) + b1
        relu1 = tf.nn.relu(fc1)

        W2 = tf.get_variable('W2', [self.config.M1, self.config.M2])
        b2 = tf.get_variable('b2', [self.config.M2])
        fc2 = tf.matmul(relu1, W2) + b2
        relu2 = tf.nn.relu(fc2)

        W3 = tf.get_variable('W3', [self.config.M2, 1])
        b3 = tf.get_variable('b3', [1])
        fc3 = tf.matmul(relu2, W3) + b3
        scores = tf.abs(fc3)

        return scores

    def add_loss_and_train_op(self, scores):
        target_scores = tf.gather(scores, self.labels)
        scores = tf.reshape(scores, (-1, self.max_as_count))
        loss = 1 + scores - target_scores
        self.loss = tf.reduce_sum(tf.reduce_max(loss * self.deltas, 1))
        optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def add_predict_op(self, scores):
        self.predictions = tf.argmax(tf.reshape(scores, (-1, self.max_as_count)), 1)

    def run_epoch(self, session, save=None, load=None):
        if not os.path.exists('./save'):
            os.makedirs('./save')
        if load:
            self.saver.restore(session, load)
        else:
            session.run(self.init_op)
        time0 = time.time()
        for epoch in range(self.config.epochs):
            time1 = time.time()
            shuffled_epoch_Rs, shuffled_epoch_HAs, shuffled_epoch_HTs, shuffled_epoch_deltas, \
                    shuffled_answer_indices = self.du.get_shuffled_data_set()
            assert len(shuffled_epoch_HTs) == len(shuffled_answer_indices) == len(shuffled_epoch_deltas)
            start_ind = 0
            len_data_set = len(shuffled_epoch_Rs)
            step = 1
            time2 = time.time()
            best_loss = float('inf')
            loss = 0
            while start_ind < len_data_set:
                time3 = time.time()
                end_ind = start_ind + self.config.batch_size
                if end_ind > len_data_set:
                    end_ind = len_data_set
                    start_ind = end_ind - self.config.batch_size
                batch_Rs = shuffled_epoch_Rs[start_ind:end_ind]
                batch_As = shuffled_epoch_HAs[start_ind:end_ind]
                batch_Ts = shuffled_epoch_HTs[start_ind:end_ind]
                batch_labels = shuffled_answer_indices[start_ind:end_ind]
                batch_deltas = shuffled_epoch_deltas[start_ind:end_ind]
                batch_HAs = self.du.encode_mention_pairs(batch_Rs, batch_Ts, batch_As)
                start_ind = end_ind
                time4 = time.time()
                batch_labels = [batch_labels[i]+self.max_as_count*i for i in range(
                                len(batch_labels))]
                feed = self.create_feed_dict(batch_HAs, batch_deltas, batch_labels)
                batch_loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed)
                time5 = time.time()
                loss += batch_loss
                if step % self.config.interval == 0:
                    print 'Epoch {}, Step {}, Time {:.2f}, Loss {:.2f}'.format(
                            epoch, step, time5-time0, batch_loss)
                step += 1

            if best_loss >= loss / step:
                self.evluation(session)
                if save is not None:
                    self.saver.save(session, save)
                else:
                    self.saver.save(session, './save/weight_{}'.format(epoch))

    def evluation(self, session, load=None):
        if load:
            self.saver.restore(session, load)

        train_answer_indices, train_h_r_antecedents = \
                self.du.get_test_data(self.config.test_batch_size, 'train')
        feed1 = self.create_feed_dict(inputs=train_h_r_antecedents)
        predictions1 = sess.run(self.predictions, feed_dict=feed1)

        test_answer_indices, test_h_r_antecedents = \
                self.du.get_test_data(self.config.test_batch_size, 'test')
        feed2 = self.create_feed_dict(inputs=test_h_r_antecedents)
        predictions2 = sess.run(self.predictions, feed_dict=feed2)

        train_acc = metrics.accuracy_score(train_answer_indices, predictions1)
        test_acc = metrics.accuracy_score(test_answer_indices, predictions2)
        
        print '============================='
        print 'Training Accuracy: {:.4f}'.format(train_acc)
        print 'Testing Accuracy: {:.4f}'.format(test_acc)
        print '============================='


if __name__ == '__main__':
    config = Config()
    cc = Coref_cluster(config)
    with tf.Session() as sess:
        cc.run_epoch(sess)

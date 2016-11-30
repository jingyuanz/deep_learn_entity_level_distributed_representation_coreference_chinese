# coding=utf-8
from compiler.ast import flatten
import polyglot
from polyglot.mapping import Embedding
from polyglot.text import Text, Word
import numpy as np
from numpy import ndarray as nd

class DataUtil:
    def __init__(self, config):
        self.config = config
        self.line_dict = {}
        self.Ts = []
        self.As = []
        self.mentions = []
        self.sentences = []
        self.sentences_t = []
        self.all_word_average = 0
        self.type_dict = {}
        self.embeddings = Embedding.load("./zh.sgns.model.tar.bz2")
        self.max_as_count = 0
        self.r_indices = []
        self.init_data()

    def get_embeddings(self):
        return self.embeddings

    def init_data(self):
        self.build_line_dict()
        self.parse_data()
        all_words = self.get_all_words()
        # print all_words
        # print self.sentences
        self.calc_word_average(all_words)

    def mention_pos(self, mention):
        line = self.sentences_t[mention[0]]
        m_count = len([word for word in line if 'n' in word or word == 't' or word=='r'])
        return [(mention[1] + 1) / m_count * 0.1]

    def distance_mentions(self, m1, m2):
        return [abs(m1[1] - m2[1])]

    def mention_equals(self, m1, m2):
        return m1[0] == m2[0] and m1[1] == m2[1] and m1[2] == m2[2] and m1[3] == m2[3]

    def distance_intervening_mentions(self, m1, m2):
        c = 0
        start = False
        for m in self.mentions:
            if self.mention_equals(m, m1):
                start = True
            if self.mention_equals(m, m2):
                break
            if start:
                c += 1
        return [c]

    def same_speaker(self, m1, m2):
        return [1]

    def is_overlap(self, m1, m2):
        if m1[2] == m2[2]:
            return [1]
        return [0]

    def get_all_words(self):
        all_words = flatten(self.sentences)
        return all_words

    def calc_word_average(self, words):
        average = sum([self.embeddings.get(word,default=np.asarray([0.0]*self.config.embedding_size)) for word in words]) / len(words) * 1.0
        return nd.tolist(average)

    def build_line_dict(self):
        with open(self.config.result_path) as f:
            lines = f.readlines()
            for line in lines:
                line_num, word_index = line.split()
                self.line_dict[int(line_num)] = int(word_index)

    def find_first_word_embedding(self, mention):
        line = self.sentences[mention[0]]
        assert line != []
        return self.embeddings.get(line[0],default=np.asarray([0.0]*self.config.embedding_size))

    def find_last_word_embedding(self, mention):
        line = self.sentences[mention[0]]
        assert line != []
        return self.embeddings.get(line[-1],default=np.asarray([0.0]*self.config.embedding_size))

    def find_following(self, mention, word_num):
        line = self.sentences[mention[0]]
        assert line != []
        word_index = mention[1]
        if word_index >= len(line) - word_num:
            for i in range(word_num - (len(line) - 1 - word_index)):
                line.append('')
        following = line[word_index + 1:word_index + word_num + 1]
        return following

    def find_proceding(self, mention, word_num):
        line = self.sentences[mention[0]]
        assert line != []
        word_index = mention[1]
        if word_index <= word_num - 1:
            for i in range(word_num - word_index):
                line = [''] + line
        proceding = line[:word_num]
        return proceding

    def average_sent(self, mention):
        line = self.sentences[mention[0]]
        assert line != []
        return self.calc_word_average(line)

    def parse_data(self):
        with open(self.config.data_path) as o_f:
            lines = o_f.readlines()
            for k in range(len(self.line_dict.items())):
                line_num, word_index = self.line_dict.items()[k]
                if word_index > -1:
                    line_mention = []
                    line = lines[line_num].decode('utf-8').split('---------->')
                    words = line[0].split()
                    self.sentences.append([word.split('/')[0] for word in words if word.split('/')[1] != 'w'])
                    self.sentences_t.append([word.split('/')[1] for word in words if word.split('/')[1] != 'w'])
                    r = line[1].strip()
                    words = [word.split('/') for word in words]
                    for i in range(len(words)):
                        w_tup = words[i]
                        if 'n' in w_tup[1] or w_tup[1] == 't' or w_tup[1]=='r':
                            mention_tup = (k, i, w_tup[0], w_tup[1])
                            self.mentions.append(mention_tup)
                            if not line_mention:
                                self.As.append([self.config.NA])
                            else:
                                self.As.append(line_mention)
                                current_len = len(line_mention)
                                if current_len>self.max_as_count:
                                    self.max_as_count = current_len
                            line_mention.append(mention_tup)
                            if w_tup[0] == r:
                                target_w = words[word_index]
                                self.r_indices.append((k,word_index))
                                target_mention_tup = (k, word_index, target_w[0], target_w[1])
                                self.Ts.append([target_mention_tup])
                            else:
                                self.Ts.append([self.config.NA])
                else:
                    self.sentences.append([])
                    self.sentences_t.append([])

    def build_feed_dict(self, start, end):
        if end > len(self.mentions):
            end = len(self.mentions)
        if start > (len(self.mentions) - self.config.batch_size):
            start = len(self.mentions) - self.config.batch_size
        return self.mentions[start:end], self.Ts[start:end], self.As[start:end]

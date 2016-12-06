# coding=utf-8
from compiler.ast import flatten
import polyglot
from polyglot.mapping import Embedding
from polyglot.text import Text, Word
import numpy as np
from numpy import ndarray as nd
import copy
import random

class DataUtil:
    def __init__(self, config):
        self.config = config
        self.line_dict = {}
        self.Ts = []
        self.As = []
        self.mentions = []
        self.sentences = []
        self.all_word_average = 0
        self.embeddings = Embedding.load("./zh.sgns.model.tar.bz2")
        self.max_as_count = 0
        self.test_rs = []
        self.test_r_answers = []
        self.test_answer_indices = []
        self.test_r_antecedents = []
        self.t_count = 0
        self.t_dict = {}
        self.r_list = []
        self.init_data()

    def get_embeddings(self):
        return self.embeddings

    def init_data(self):
        self.build_line_dict()
        self.parse_data()
        self.compute_r_a_tuples()
        all_words = self.get_all_words()
        # print all_words
        # print self.sentences
        self.calc_word_average(all_words)

    def mention_pos(self, mention):
        line = self.sentences[mention[0]]
        m_count = len([word[0] for word in line if word and ('n' in word[1] or word[1] in self.config.mention_types)])
        return [(mention[1] + 1) / m_count * 0.1]

    def distance_mentions(self, m1, m2):
        # [0,1,2,3,4,5-7,8-15,16-31,32-63,64+]
        d = abs(m2[1] - m1[1])
        if d == 0:
            return [1,0,0,0,0,0,0,0,0,0]
        elif d==1:
            return [0,1,0,0,0,0,0,0,0,0]
        elif d==2:
            return [0,0,1,0,0,0,0,0,0,0]
        elif d==3:
            return [0,0,0,1,0,0,0,0,0,0]
        elif d==4:
            return [0,0,0,0,1,0,0,0,0,0]
        elif d>=5 and d<=7:
            return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif d>=8 and d<=15:
            return [0,0,0,0,0,0,1,0,0,0]
        elif d>=16 and d<=31:
            return [0,0,0,0,0,0,0,1,0,0]
        elif d>=32 and d<=63:
            return [0,0,0,0,0,0,0,0,1,0]
        else:
            return [0,0,0,0,0,0,0,0,0,1]

    def mention_equals(self, m1, m2):
        return m1[0] == m2[0] and m1[1] == m2[1] and m1[2] == m2[2] and m1[3] == m2[3]

    def distance_intervening_mentions(self, m1, m2):
        d = 0
        start = False
        for m in self.mentions:
            if self.mention_equals(m, m1):
                start = True
            if self.mention_equals(m, m2):
                break
            if start:
                d += 1

        if d == 0:
            return [1,0,0,0,0,0,0,0,0,0]
        elif d==1:
            return [0,1,0,0,0,0,0,0,0,0]
        elif d==2:
            return [0,0,1,0,0,0,0,0,0,0]
        elif d==3:
            return [0,0,0,1,0,0,0,0,0,0]
        elif d==4:
            return [0,0,0,0,1,0,0,0,0,0]
        elif d>=5 and d<=7:
            return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif d>=8 and d<=15:
            return [0,0,0,0,0,0,1,0,0,0]
        elif d>=16 and d<=31:
            return [0,0,0,0,0,0,0,1,0,0]
        elif d>=32 and d<=63:
            return [0,0,0,0,0,0,0,0,1,0]
        else:
            return [0,0,0,0,0,0,0,0,0,1]

    def same_speaker(self, m1, m2):
        return [1]

    def is_overlap(self, m1, m2):
        if m1[2] == m2[2]:
            return [1]
        return [0]

    def get_all_words(self):
        all_words = []
        for sent in self.sentences:
            if sent:
                all_words += sent
        return all_words

    def calc_word_average(self, words):
        words = [word for word in words if word != '']
        if len(words) == 0:
            return [np.float32(0.0)]*self.config.embedding_size
        average = sum([self.embeddings.get(word[0], word[1], default=np.asarray([np.float32(0.0)]*self.config.embedding_size)) for word in words]) / len(words)
        return nd.tolist(average)

    def build_line_dict(self):
        with open(self.config.result_path) as f:
            lines = f.readlines()
            for line in lines:
                line_num, word_index = line.strip().split()
                self.line_dict[int(line_num)] = int(word_index)

    def find_first_word_embedding(self, mention):
        line = self.sentences[mention[0]]
        assert line != []
        return self.embeddings.get(line[0][0], line[0][1], default=np.asarray([0.0]*self.config.embedding_size))

    def find_last_word_embedding(self, mention):
        line = self.sentences[mention[0]]
        assert line != []
        return self.embeddings.get(line[-1][0], line[-1][1], default=np.asarray([0.0]*self.config.embedding_size))

    def find_following(self, mention, word_num):
        line = copy.copy(self.sentences[mention[0]])
        assert line != []
        word_index = mention[1]
        for i in range(word_num):
            line.append('')
        following = line[word_index + 1:word_index + word_num + 1]
        return following

    def find_proceding(self, mention, word_num):
        line = copy.copy(self.sentences[mention[0]])
        assert line != []
        word_index = mention[1]
        for i in range(word_num):
            line = [''] + line
        proceding = line[word_index:word_index+word_num]
        return proceding


    def find_following_embeddings(self, mention, word_num):

        line = copy.copy(self.sentences[mention[0]])
        assert line != []
        word_index = mention[1]
        for i in range(word_num):
            line.append('None')
        following = line[word_index + 1:word_index + word_num + 1]
        follow_embed = []
        for follow in following:
            if follow == "None":
                follow_embed.append([0.0]*self.config.embedding_size)
            else:
                follow_embed.append(nd.tolist(self.embeddings.get(follow[0], follow[1], default=np.asarray([0.0]*self.config.embedding_size))))
        # print follow_embed, flatten(follow_embed)
        return flatten(follow_embed)

    def find_proceding_embeddings(self, mention, word_num):
        line = copy.copy(self.sentences[mention[0]])
        assert line != []
        word_index = mention[1]
        for i in range(word_num):
            line = ['None'] + line
        proceding = line[word_index:word_index+word_num]
        proced_embed = []
        for proced in proceding:
            if proced == "None":
                proced_embed.append([0.0]*self.config.embedding_size)
            else:
                proced_embed.append(nd.tolist(self.embeddings.get(proced[0], proced[1], default=np.asarray([0.0]*self.config.embedding_size))))
        # print proced_embed, flatten(proced_embed)
        # assert mention[2] in [word[0] for word in line if word!="None"]

        # print "line: ", ''.join([word[0] for word in line if word!="None"])
        # print "origin: ", line
        # print "len: ", len(line)
        # print "mention: ", mention[2]
        # print "index: ", mention[0], mention[1]
        # print "proceding: ", proceding
        # print "embed: ", proced_embed
        # print "len_embed", len(proced_embed)
        # print
        return flatten(proced_embed)

    def average_sent(self, mention):
        line = self.sentences[mention[0]]
        assert line != []
        return self.calc_word_average(line)

    def parse_data(self):
        with open(self.config.data_path) as o_f:
            lines = o_f.readlines()
            self.lines = lines
            for line in lines:
                line = line.decode('utf-8').strip().split('---------->')
                words = line[0].split()
                r = line[1].strip()
                tups = [tuple(word.split('/')) for word in words]
                target_r = ''
                for tup_i in range(len(tups)):
                    if tups[tup_i][1] == 'r' and tups[tup_i][0] == r:
                        target_r = (tups[tup_i][0], tup_i)
                assert target_r
                self.r_list.append(target_r)
                self.sentences.append(tups)

            for line_num, word_index in self.line_dict.items():
                # print line_num, word_index
                # if line_num==8198:
                #     for w_8198 in self.sentences[line_num]:
                #         print w_8198[0],
                #     print self.r_list[8198], word_index, len(self.sentences[line_num])
                if word_index > -1:
                    line_mention = []
                    words = self.sentences[line_num]
                    r = self.r_list[line_num]
                    for i in range(len(words)):
                        w_tup = words[i]
                        word_type = w_tup[1]
                        if not self.t_dict.has_key(word_type):
                            self.t_dict[word_type] = self.t_count
                            self.t_count += 1
                        if 'n' in w_tup[1] or w_tup[1] in self.config.mention_types:
                            mention_tup = (line_num, i, w_tup[0], w_tup[1])
                            self.mentions.append(mention_tup)
                            if not line_mention:
                                self.As.append([self.config.NA])
                            else:
                                self.As.append(line_mention)
                                current_len = len(line_mention)
                                if current_len>self.max_as_count:
                                    self.max_as_count = current_len
                            if w_tup[0] == r[0] and i == r[1]:
                                target_w = words[word_index]
                                self.test_rs.append(mention_tup)
                                target_mention_tup = (line_num, word_index, target_w[0], target_w[1])
                                # print target_mention_tup
                                self.test_r_answers.append(target_mention_tup)
                                self.test_r_antecedents.append(line_mention)
                                assert len(self.test_r_answers) == len(self.test_r_antecedents) == len(self.test_rs)
                                self.Ts.append(target_mention_tup)
                            else:
                                self.Ts.append(self.config.NA)
                            line_mention.append(mention_tup)

    def build_feed_dict(self, start, end):
        if end > len(self.mentions):
            end = len(self.mentions)
        if start > (len(self.mentions) - self.config.batch_size):
            start = len(self.mentions) - self.config.batch_size
        return self.mentions[start:end], self.Ts[start:end], self.As[start:end]

    def compute_r_a_tuples(self):
        for i in range(len(self.test_rs)):
            r = self.test_rs[i]
            ans = self.test_r_answers[i]
            found = False
            # self.test_r_answers[i] = (self.test_r_answers[i],r)
            for k in range(len(self.test_r_antecedents[i])):
                test_ante = self.test_r_antecedents[i][k]

                if test_ante!=self.config.NA and self.mention_equals(test_ante, ans):
                    found = True
                    # print test_ante
                    self.test_answer_indices.append(k)
            if not found:
                print r[0], r[1], r[2]
                print ans[0], ans[1], ans[2], ans[3]



            self.test_r_antecedents[i] = map(lambda x:(x,r),self.test_r_antecedents[i])
            padding = [(self.config.NA,r)]
            self.test_r_antecedents[i].extend(padding*(self.max_as_count+1-len(self.test_r_antecedents[i])))
            # print self.test_r_answers[i][0][2], self.test_r_answers[i][1][2]
            # print len(self.test_r_antecedents[i]), len(self.test_r_antecedents[i][0])

    def get_test_data(self, size):
        # return self.test_r_answers[:size], self.test_r_antecedents[:size]
        return self.test_answer_indices[-size:], self.test_r_antecedents[-size:]
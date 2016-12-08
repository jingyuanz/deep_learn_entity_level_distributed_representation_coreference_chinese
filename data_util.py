# coding=utf-8
from compiler.ast import flatten
import polyglot
from polyglot.mapping import Embedding
from polyglot.text import Text, Word
import numpy as np
from numpy import ndarray as nd
import copy
import sys
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
        m_count = len([word[0] for word in line if word and ('n' in word[1] or word[1] == 't')])
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
                target_mention_tup = (line_num, word_index, self.sentences[line_num][word_index][0], self.sentences[line_num][word_index][1])
                if word_index > -1 and ('n' in target_mention_tup[3] or target_mention_tup[3] == 't') and len(self.sentences[line_num]) <= 50:
                    line_mention = []
                    words = self.sentences[line_num]
                    r = self.r_list[line_num]
                    for i in range(len(words)):
                        w_tup = words[i]
                        word_type = w_tup[1]
                        word_span = w_tup[0]
                        if not self.t_dict.has_key(word_type):
                            self.t_dict[word_type] = self.t_count
                            self.t_count += 1
                        if 'n' in word_type or word_type == 't':
                            mention_tup = (line_num, i, word_span, word_type)
                            self.mentions.append(mention_tup)
                            line_mention.append(mention_tup)
                            # if not line_mention:
                            #     self.As.append([self.config.NA])
                            # else:
                            #     self.As.append(copy.copy(line_mention))
                            #     current_len = len(line_mention)
                            #     if current_len>self.max_as_count:
                            #         self.max_as_count = current_len
                        elif word_span == r[0] and i == r[1]:
                            r_tup = (line_num, i, word_span, 'r')
                            # print "r->",r_tup, r_tup[2]
                            if target_mention_tup[1] < i:
                                self.test_rs.append(r_tup)
                                # print target_mention_tup
                                self.test_r_answers.append(target_mention_tup)
                                # print "target->", target_mention_tup, target_mention_tup[2]
                                if line_mention:
                                    if len(line_mention)>self.max_as_count:
                                        self.max_as_count = len(line_mention)
                                    self.test_r_antecedents.append(copy.copy(line_mention))
                                else:
                                    print "shud not reach here", line_num
                                    self.test_r_antecedents.append([self.config.NA])
                                assert len(self.test_r_answers) == len(self.test_r_antecedents) == len(self.test_rs)

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



            # self.test_r_antecedents[i] = map(lambda x:(x,r),self.test_r_antecedents[i])
            padding = [self.config.NA]
            self.test_r_antecedents[i].extend(padding*(self.max_as_count+1-len(self.test_r_antecedents[i])))
            # print self.test_r_answers[i][0][2], self.test_r_answers[i][1][2]
            # print len(self.test_r_antecedents[i]), len(self.test_r_antecedents[i][0])

    def h(self, a, m):
        if a == 0 and m == 0:
            result = [np.float32(0.0)]*self.config.I
            return result
        if a=='#':
            a = m
        embed_a = nd.tolist(self.embeddings.get(a[2],a[3],default=np.asarray([0.0]*self.config.embedding_size)))
        embed_m = nd.tolist(self.embeddings.get(m[2],m[3],default=np.asarray([0.0]*self.config.embedding_size)))
        # print len(embed_m)
        first_aw_embed = nd.tolist(self.find_first_word_embedding(a))
        # print len(first_aw_embed)
        first_mw_embed = nd.tolist(self.find_first_word_embedding(m))
        # print len(first_mw_embed)
        last_aw_embed = nd.tolist(self.find_last_word_embedding(a))
        # print len(last_aw_embed)
        last_mw_embed = nd.tolist(self.find_last_word_embedding(m))
        # print len(last_mw_embed)
        proced2_a_embed = self.find_proceding_embeddings(a, 2)
        follow2_a_embed = self.find_following_embeddings(a, 2)

        proced2_m_embed = self.find_proceding_embeddings(m, 2)
        follow2_m_embed = self.find_following_embeddings(m, 2)

        avg5f_a = self.calc_word_average(self.find_following(a, 5))
        # print len(avg5f_a)
        avg5p_a = self.calc_word_average(self.find_proceding(a, 5))
        # print len(avg5p_a)
        avg5f_m = self.calc_word_average(self.find_following(m, 5))
        # print len(avg5f_m)
        avg5p_m = self.calc_word_average(self.find_proceding(m, 5))
        # print len(avg5p_m)
        avgsent_a = self.average_sent(a)
        # print len(avgsent_a)
        avgsent_m = self.average_sent(m)
        # print len(avgsent_m)
        avg_all = [self.all_word_average]
        # print len(avg_all)
        type_a = [self.t_dict[a[3]]]  # self.type_dict[a[3]]
        type_m = [self.t_dict[m[3]]]  # self.type_dict[m[3]]
        mention_pos_a = self.mention_pos(a)
        mention_pos_m = self.mention_pos(m)

        mention_len_a = [len(a[2])]
        mention_len_m = [len(m[2])]

        distance = self.distance_mentions(a, m)
        distance_m = self.distance_intervening_mentions(a, m)

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

    def get_test_data(self, size, mode):
        if mode=='test':
        # return self.test_r_answers[:size], self.test_r_antecedents[:size]
            rs_batch = self.test_rs[-size:]
            r_answers = self.test_answer_indices[-size:]
            r_antecedents = self.test_r_antecedents[-size:]
        # return self.test_answer_indices, self.test_r_antecedents
        else:
            rs_batch = self.test_rs[:size]
            r_answers = self.test_answer_indices[:size]
            r_antecedents = self.test_r_antecedents[:size]

        h_r_antecedents = []
        for combo_i in range(len(rs_batch)):
            combo_r = rs_batch[combo_i]
            combo_as = r_antecedents[combo_i]
            combos = [self.h(combo_a, combo_r) for combo_a in combo_as]
            h_r_antecedents.append(combos)

        return rs_batch, r_answers, h_r_antecedents





    def mistake(self, a, T):
        if a == self.config.NA and T != self.config.NA:
            return self.config.a_fn
        if a != self.config.NA and T == self.config.NA:
            return self.config.a_fa
        if a != self.config.NA and a != T:
            return self.config.a_wl
        if a == T:
            return 0

    def encode_mention_pairs(self, batch_Rs, batch_Ts, batch_As):
        batch_HTs = []
        for j in range(len(batch_Ts)):
            t = batch_Ts[j]
            r = batch_Rs[j]
            ht = self.h(t, r)
            batch_HTs.append(ht)
        # hts = np.array(hts)
        batch_HAs = []
        for z in range(len(batch_Rs)):
            As = batch_As[z]
            r = batch_Rs[z]
            HA = [self.h(a, r) for a in As]
            padding = [np.float32(0.0)] * self.config.I
            HA.extend([padding] * (self.max_as_count - len(HA)))
            batch_HAs.append(HA)
        return batch_HAs, batch_HTs

    def get_shuffled_data_set(self):

        random_indices = [randi for randi in range(len(self.test_rs) - self.config.test_batch_size)]
        random.shuffle(random_indices)
        Rs = []
        As = []
        Ts = []
        Ans_indices = []
        for rand_i in random_indices:
            Rs.append(self.test_rs[rand_i])
            As.append(self.test_r_antecedents[rand_i])
            Ts.append(self.test_r_answers[rand_i])
            Ans_indices.append(self.test_answer_indices[rand_i])

        mistakes = []

        for k in range(len(Ts)):
            T = Ts[k]
            A = As[k]
            mistake = [np.float32(self.mistake(a, T)) for a in A]
            mistake.extend([np.float32(0.0)] * (self.max_as_count - len(mistake)))
            mistakes.append(mistake)

        return Rs, As, Ts, mistakes, Ans_indices



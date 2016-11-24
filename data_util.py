#coding=utf-8
from compiler.ast import flatten


class DataUtil:
    def __init__(self, config):
        self.config = config
        self.line_dict = {}
        self.Ts = []
        self.As = []
        self.mentions = []
        self.sentences = []
        self.all_word_average = 0
        self.type_dict = {}

    def mention_pos(self, mention):
        line = self.sentences[mention[0]]
        m_count = len([word for word in line if 'n' in word or word == 't'])
        return (mention[1]+1)/m_count

    def distance_mentions(self, m1, m2):
        return abs(m1[1]-m2[1])

    def is_overlap(self, m1, m2):
        if m1[2]==m2[2]:
            return 1
        return 0

    def calc_all_word_average(self):
        all_words = flatten(self.sentences)
        #TODO

    def build_line_dict(self):
        with open('results.txt') as f:
            lines = f.readlines()
            for line in lines:
                line_num, word_index = line.split()
                self.line_dict[int(line_num)] = int(word_index)

    def find_first_word(self, mention):
        line_num = mention[0]
        return self.sentences[line_num][0]

    def find_last_word(self, mention):
        line_num = mention[0]
        return self.sentences[line_num][-1]

    def find_following(self, mention, word_num):
        line= self.sentences[mention[0]]
        word_index = mention[1]
        if word_index >= len(line)-word_num:
            for i in range(word_num-(len(line)-1-word_index)):
                line.append('')
        following = line[word_index+1:word_index+word_num+1]
        return following

    def find_proceding(self, mention, word_num):
        line = self.sentences[mention[0]]
        word_index = mention[1]
        if word_index <= word_num - 1:
            for i in range(word_num-word_index):
                line = ['']+line
        proceding = line[:word_num]
        return proceding

    def average_sent(self, mention):
        line = self.sentences[mention[0]]
        #TODO

    def read_feed(self):
        with open('out.txt') as o_f:
            lines = o_f.readlines()
            for k in range(len(self.line_dict.items())):
                line_num, word_index = self.line_dict.items()[k]
                if word_index > -1:
                    line_mention = []
                    line = lines[line_num].decode('utf-8').split('---------->')
                    words = line[0].split()
                    self.sentences.append([word.split('/')[0] for word in words if word.split('/')[1] != 'w'])
                    r = line[1].strip()
                    words = [word.split('/') for word in words]
                    for i in range(len(words)):
                        w_tup = words[i]
                        if 'n' in w_tup[1] or w_tup[1] == 't':
                            self.mentions.append((k, i, w_tup[0], w_tup[1]))
                            if not line_mention:
                                self.As.append([self.config.NA])
                            else:
                                self.As.append(line_mention)
                            line_mention.append(w_tup[0])
                            if w_tup[0] == r:
                                self.Ts.append([words[word_index]])
                            else:
                                self.Ts.append([self.config.NA])
                else:
                    self.sentences.append([])









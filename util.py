from collections import Counter, defaultdict
from itertools import count
from collections import defaultdict
from itertools import count, izip, chain
# import itertools
import operator
import random

import dynet as dy
import numpy as np
import sys
import time


class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.iteritems()}

    @classmethod
    def from_corpus(cls, corpus, top=20000):
        freqs = Counter(chain(*corpus))
        # print len(freqs)
        sorted_freqs = sorted(freqs.iteritems(), key=operator.itemgetter(1), reverse=True)

        w2i = defaultdict(count(0).next)
        w2i["<unk>"]
        w2i["<s>"]
        w2i["</s>"]
        # print "w2i for <unk> <s> </s> " + str(w2i["<unk>"]) + str(w2i["<s>"]) + str(w2i["</s>"])

        [w2i[key] for key, value in sorted_freqs[:top - 1] if value > 1]  # eliminate singleton


        vocab=Vocab(w2i)

        return vocab

    def size(self):
        return len(self.w2i.keys())

    def word2Wid(self, word):
        if word in self.w2i:
            return self.w2i[word]
        else:
            return self.w2i["<unk>"]


def get_data_id(src_vocab, data):
    data_id = []
    for sent in data:

        data_id.append([src_vocab.word2Wid(word) for word in sent])

    return data_id



def read_corpus(fname):
    data=[]
    with file(fname) as fh:
        for line in fh:
            sent = line.strip().split()
            data.append(["<s>"] + sent + ["</s>"])
            # yield ["<s>"] + sent + ["</s>"]
    return data


class Hypothesis(object):
    def __init__(self, state, y, ctx_tm1, score):
        self.state = state
        self.y = y
        self.ctx_tm1 = ctx_tm1
        self.score = score

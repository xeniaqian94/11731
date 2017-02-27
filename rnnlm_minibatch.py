from collections import Counter, defaultdict
from itertools import count
import random

import dynet as dy
import numpy as np
import sys
import time

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file = sys.argv[1]
test_file = sys.argv[2]

MB_SIZE = 32

N = 3
EVAL_EVERY = 10000
EMB_SIZE = 128
HID_SIZE = 128
NUM_EPOCHES=20


class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.iteritems()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self):
        return len(self.w2i.keys())

    def word2Wid(self,word):
        if word in self.w2i:
            return self.w2i[word]
        else:
            return self.w2i["<unk>"]


def read(fname):
    with file(fname) as fh:
        for line in fh:
            sent = line.strip().split()
            sent.append("<s>")
            yield sent


train = list(read(train_file))
test = list(read(test_file))
words = []
wc = Counter()
for sent in train:
    for w in sent:
        words.append(w)
        wc[w] += 1
words.append("<unk>")

vw = Vocab.from_corpus([words])
S = vw.w2i["<s>"]

nwords = vw.size()

# DyNet Starts

model = dy.Model()
trainer = dy.SimpleSGDTrainer(model)

# Lookup parameters for word embeddings
WORDS_LOOKUP = model.add_lookup_parameters((nwords, EMB_SIZE))

# Word-level LSTM (layers=1, input=64, output=128, model)
RNN = dy.LSTMBuilder(1, EMB_SIZE, HID_SIZE, model)

# Softmax weights/biases on top of LSTM outputs
W_sm = model.add_parameters((nwords, HID_SIZE))
b_sm = model.add_parameters(nwords)


# Build the language model graph
def calc_lm_loss(sents):
    dy.renew_cg()
    # parameters -> expressions
    W_exp = dy.parameter(W_sm)
    b_exp = dy.parameter(b_sm)

    # initialize the RNN
    f_init = RNN.initial_state()

    # get the wids and masks for each step
    tot_words = 0
    wids = []
    masks = []
    # print "len sents[0]"+str(len(sents[0])) +" number within this batch "+str(len(sents))
    for i in range(len(sents[0])):
        wids.append([(vw.word2Wid(sent[i]) if len(sent) > i else S) for sent in sents])
        mask = [(1 if len(sent) > i else 0) for sent in sents]
        # print "len of mask "+str(len(mask))
        masks.append(mask)
        tot_words += sum(mask)

    # start the rnn by inputting "<s>"
    init_ids = [S] * len(sents)
    s = f_init.add_input(dy.lookup_batch(WORDS_LOOKUP, init_ids))

    # feed word vectors into the RNN and predict the next word
    losses = []
    for wid, mask in zip(wids, masks):
        # calculate the softmax and loss
        score = W_exp * s.output() + b_exp
        loss = dy.pickneglogsoftmax_batch(score, wid)
        # mask the loss if at least one sentence is shorter
        if mask[-1] != 1:
            mask_expr = dy.inputVector(mask)
            # print len(mask)
            mask_expr = dy.reshape(mask_expr, (1,), len(mask))
            loss = loss * mask_expr
        losses.append(loss)
        # update the state of the RNN
        wemb = dy.lookup_batch(WORDS_LOOKUP, wid)
        s = s.add_input(wemb)

    return dy.sum_batches(dy.esum(losses)), tot_words


num_tagged = cum_loss = 0
# Sort training sentences in descending order and count minibatches
train.sort(key=lambda x: -len(x))
test.sort(key=lambda x: -len(x))
print "length of train "+str(len(train))+" num of batches "+str(len(train) / MB_SIZE + 1)
print "length of test "+str(len(test))+" num of batches "+str(len(test) / MB_SIZE + 1)
train_order = [x * MB_SIZE for x in range(len(train) / MB_SIZE + 1)]
test_order = [x * MB_SIZE for x in range(len(test) / MB_SIZE + 1)]
# Perform training
start_time=time.time()

for ITER in xrange(NUM_EPOCHES):
    random.shuffle(train_order)
    for i, sid in enumerate(train_order, 1):
        if i % (EVAL_EVERY / MB_SIZE) == 0:
            # trainer.status()
            print cum_loss / num_tagged
            num_tagged = cum_loss = 0
        if i % (EVAL_EVERY / MB_SIZE) == 0 or i == len(train_order) - 1:
            dev_loss = dev_words = 0
            for sid in test_order:
                loss_exp, mb_words = calc_lm_loss(test[sid:sid + MB_SIZE])
                dev_loss += loss_exp.scalar_value()
                dev_words += mb_words
            print "Epoch=%d Updates=%d PPL=%f Time=%d" %(ITER,i,np.exp(dev_loss / dev_words),time.time()-start_time)
        # train on the minibatch
        loss_exp, mb_words = calc_lm_loss(train[sid:sid + MB_SIZE])
        cum_loss += loss_exp.scalar_value()
        num_tagged += mb_words
        loss_exp.backward()
        trainer.update()
        print "trainer updated once "+str(i)+" "+str(sid)
    print "epoch %r finished" % ITER
    trainer.update_epoch(1.0)
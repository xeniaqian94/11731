import sys
import random
import math
import dynet as dy
from collections import defaultdict
import numpy as np
import time
import os

# Define the hyperparameters
N = 3
EVAL_EVERY = 10000
EMB_SIZE = 128
HID_SIZE = 128
NUM_EPOCHES=20

# perplexity baseline = 200

# Loop over the words, counting the frequency
# open the training file
# count the frequency of words

# the same as the previous assignment
word_frequencies = dict()

file = open(sys.argv[1], "r")
word_frequencies = dict()
lines = file.readlines()
for line in lines:
    words = line.strip().split()
    for word in words:
        value = word_frequencies.setdefault(word, 0)
        word_frequencies[word] = value + 1


# Read in the training and validation data and parse it into context/word pairs
def create_data(fname):
    # the same as the previous assignment
    data=[]
    file = open(fname, "r")
    lines = file.readlines()
    length=0
    for line in lines:
        length+=len(line.strip().split())
        words = ["</s>"] + line.strip().split() + ["<s>"]
        for i in range(2,len(words)):
            data+=[((words[i-2],words[i-1]),words[i])]
    return data,length


train_data,length = create_data(sys.argv[1])

# Create the word vocabulary for all words > 1
# loop over the words, and create word IDs for all words < 1
wids = defaultdict(lambda: 0)  # cutoff to be implemented
wids["<unk>"] = 0
wids["<s>"] = 1
wids["</s>"] = 2
for word, freq in word_frequencies.items():
    if freq > 3:
        wids[word] = len(wids)

VOCAB_SIZE=len(wids)

def word2Wid(word):
    if word in wids:
        return wids[word]
    else:
        return wids["<unk>"]

# Define the model and SGD optimizer
model = dy.Model()
lookup = model.add_lookup_parameters((len(wids), EMB_SIZE))
W_mh_p = model.add_parameters((HID_SIZE, EMB_SIZE * (N - 1)))
b_h_p = model.add_parameters((HID_SIZE))
W_hs_p = model.add_parameters((len(wids), HID_SIZE))
b_s_p = model.add_parameters((len(wids)))
trainer=dy.SimpleSGDTrainer(model)

# The function to calculate the loss
def calc_loss(ctxt, word):
    y=calc_function(ctxt)
    loss=dy.pickneglogsoftmax(y,word2Wid(word))
    return loss

def calc_function(ctxt):
    dy.renew_cg()
    # m=dy.lookup(lookup,wids[ctxt[0]]).value()+dy.lookup(lookup,wids[ctxt[1]]).value()
    m_val=dy.concatenate([dy.lookup(lookup,wids[ctxt[0]]),dy.lookup(lookup,wids[ctxt[1]])])

    W_mh = dy.parameter(W_mh_p)
    b_h = dy.parameter(b_h_p)
    W_hs = dy.parameter(W_hs_p)
    b_s = dy.parameter(b_s_p)

    h_val=dy.tanh(W_mh*m_val+b_h)
    y_val=W_hs*h_val+b_s
    y_val=dy.softmax(y_val)
    # print y_val
    return y_val

validation_data,length = create_data(sys.argv[2])


def eval_dev(validation_data,length):
    sumLog=0
    for (ctxt,word) in validation_data:
        y=calc_function(ctxt).value()
        prob=y[word2Wid(word)]
        sumLog+=np.log(prob)

    print "test corpus length "+str(length)
    ppl=np.exp(-1.0*sumLog/length)
    print ppl #perplexity
    return ppl

for epoch in range(NUM_EPOCHES):
    epoch_loss=[]
    random.shuffle(train_data)
    train_length=len(train_data)
    i=0
    for (ctxt,word) in train_data:
        i=i+1
        if i%1000==10:
            ppl = eval_dev(validation_data, length)
            # print ("Training set Epoch %d: loss=%f" % (epoch, dy.esum(epoch_loss)))
            print i,train_length,time.time()
        loss=calc_loss(ctxt,word)
        # epoch_loss.append(loss)
        loss.backward()
        trainer.update()








# looking up embedding
# basically the same as the example, but we use the lookup and softmax instead

# For data in the training set, train. Evaluate dev set occasionally
# basically the same as the log-linear model

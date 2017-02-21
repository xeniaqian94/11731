import sys
import random
import math
import dynet as dy
from collections import defaultdict

# Define the hyperparameters
N = 3
EVAL_EVERY = 10000
EMB_SIZE = 128
HID_SIZE = 128

# Loop over the words, counting the frequency
# open the training file
# count the frequency of words

# Create the word vocabulary for all words > 1
# loop over the words, and create word IDs for all words < 1
wids = defaultdict(lambda: 0)
wids["<unk>"] = 0
wids["<s>"] = 1
wids["</s>"] = 2
for word, freq in word_frequencies.items():
  if freq > 3:
    wids[word] = len(wids) 

# Read in the training and validation data and parse it into context/word pairs
def create_data(fname):
  # the same as the previous assignment

train_data = create_data(sys.argv[1])
valid_data = create_data(sys.argv[2])

# Create the neural network model including lookup parameters, etc
model = dy.Model()
M = dy.add_lookup_parameters((len(wids), EMB_SIZE))
W_mh = dy.add_parameters((HID_SIZE, EMB_SIZE * (N-1)))
b_h = dy.add_parameters((HID_SIZE))
W_hs = dy.add_parameters((len(wids), HID_SIZE))
b_s = dy.add_parameters((len(wids)))

# The function to calculate the loss
def calc_loss(ctxt, wid):
  # basically the same as the example, but we use the lookup and softmax instead

# For data in the training set, train. Evaluate dev set occasionally
# basically the same as the log-linear model
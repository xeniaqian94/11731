import sys
import dynet as dy
import numpy as np
import argparse
from dynet import *
import copy

from nltk.translate.bleu_score import corpus_bleu
from collections import Counter, defaultdict
from collections import defaultdict
from itertools import count, izip, chain
import operator
import time


class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.iteritems()}

    @classmethod
    def from_corpus(cls, corpus, top=20000):
        freqs = Counter(chain(*corpus))
        # print len(freqs)class Vocab:

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

        [w2i[key] for key, value in sorted_freqs[:top - 3] if value > 1]  # eliminate singleton
        vocab = Vocab(w2i)
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
    data = []
    with file(fname) as fh:
        for line in fh:
            sent = line.strip().split()
            data.append(["<s>"] + sent + ["</s>"])
            # yield ["<s>"] + sent + ["</s>"]
    return data


def get_batches(sents_pair, batch_size):
    buckets = defaultdict(list)
    [buckets[len(pair[0])].append(pair) for pair in sents_pair]
    batches = []

    for length in buckets:
        bucket = buckets[length]
        np.random.shuffle(bucket)
        for i in range(int(np.ceil(len(bucket) * 1.0 / batch_size))):
            elements_count = min(batch_size, len(bucket) - batch_size * i)
            batches.append(([bucket[i * batch_size + j][0] for j in range(elements_count)],
                            [bucket[i * batch_size + j][1] for j in range(elements_count)]))

    np.random.shuffle(batches)
    return batches


class EncoderDecoder:
    def __init__(self, args, src_vocab, tgt_vocab, src_id_to_token, tgt_id_to_token):

        self.model = model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        self.args = args
        self.src_vocab, self.src_token_to_id, self.src_id_to_token = src_vocab, src_vocab.w2i, src_vocab.i2w
        self.src_vocab_size = self.src_vocab.size()
        self.tgt_vocab, self.tgt_token_to_id, self.tgt_id_to_token = tgt_vocab, tgt_vocab.w2i, tgt_vocab.i2w
        self.tgt_vocab_size = self.tgt_vocab.size()

        self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.args.emb_size))
        self.tgt_lookup = model.add_lookup_parameters((self.tgt_vocab_size, self.args.emb_size))
        self.l2r_builder = GRUBuilder(1, self.args.emb_size, self.args.hid_dim, model)
        self.r2l_builder = GRUBuilder(1, self.args.emb_size, self.args.hid_dim, model)

        self.W_init = model.add_parameters((self.args.hid_dim, self.args.hid_dim * 2))
        self.b_init = model.add_parameters((self.args.hid_dim, 1))
        self.b_init.zero()
        self.dec_builder = GRUBuilder(1, self.args.emb_size + self.args.hid_dim * 2, self.args.hid_dim, model)

        self.W_logit_cxt = model.add_parameters((self.args.emb_size, self.args.hid_dim * 2))
        self.W_logit_input = model.add_parameters((self.args.emb_size, self.args.emb_size))
        self.W_logit_hid = model.add_parameters((self.args.emb_size, self.args.hid_dim))
        self.b_logit_readout = model.add_parameters((self.args.emb_size, 1))
        self.b_logit_readout.zero()

        self.W1_att_e = model.add_parameters((self.args.att_dim, self.args.hid_dim))
        self.W1_att_f = model.add_parameters((self.args.att_dim, self.args.hid_dim * 2))
        self.w2_att = model.add_parameters((1, self.args.att_dim))

        self.softmax_W = model.add_parameters((self.tgt_vocab_size, self.args.emb_size))
        self.softmax_b = model.add_parameters((self.tgt_vocab_size,))
        self.softmax_b.zero()

    def save(self):
        self.model.save(
            "./model/embed_" + str(self.args.emb_size) + "_hidden_" + str(self.args.hid_dim) + "_attn_" + str(
                self.args.att_dim))

    def load(self, model_name=None):
        if model_name:
            self.model.load("./model/" + model_name)
        else:
            self.model.load(
                "./model/embed_" + str(self.args.emb_size) + "_hidden_" + str(self.args.hid_dim) + "_attn_" + str(
                    self.args.att_dim))

    def transpose_input(self, src_sents):
        wids = []
        masks = []
        for i in range(max([len(sent) for sent in src_sents])):
            wids.append([sent[i] if len(sent) > i else 2 for sent in src_sents])  # pad </s>
            masks.append([1 if len(sent) > i else 0 for sent in src_sents])
        return wids, masks

    def encode(self, src_sents):
        dy.renew_cg()
        wids, masks = self.transpose_input(src_sents)
        l2r_wid_embeds = [dy.lookup_batch(self.src_lookup, wid) for wid in wids]
        l2r_encodings = self.l2r_builder.initial_state().transduce(l2r_wid_embeds)
        r2l_encodings = self.r2l_builder.initial_state().transduce(list(reversed(l2r_wid_embeds)))
        r2l_encodings = list(reversed(r2l_encodings))

        return [dy.concatenate(
            [l2r_encoding, r2l_encoding]) for (l2r_encoding, r2l_encoding) in zip(l2r_encodings, r2l_encodings)]

    def attention(self, encoding, hidden, batch_size):

        W1_att_f = dy.parameter(self.W1_att_f)
        W1_att_e = dy.parameter(self.W1_att_e)
        w2_att = dy.parameter(self.w2_att)

        H = dy.concatenate_cols(encoding)
        a = dy.softmax(dy.reshape(
            w2_att * dy.tanh(dy.colwise_add(W1_att_f * H, W1_att_e * hidden)), (len(encoding),),
            batch_size))  # equation 81

        return H * a, a

    def decode_loss(self, encoding, tgt_sents):
        batch_size = len(tgt_sents)
        maxLen = max(len(tgt_sent) for tgt_sent in tgt_sents)

        W_init = dy.parameter(self.W_init)
        b_init = dy.parameter(self.b_init)

        W_logit_cxt = dy.parameter(self.W_logit_cxt)
        W_logit_input = dy.parameter(self.W_logit_input)
        W_logit_hid = dy.parameter(self.W_logit_hid)
        b_logit_readout = dy.parameter(self.b_logit_readout)

        W_y = dy.parameter(self.softmax_W)
        b_y = dy.parameter(self.softmax_b)

        dec_state = self.dec_builder.initial_state([dy.tanh(dy.affine_transform([b_init, W_init, encoding[-1]]))])

        tgt_wids, tgt_masks = self.transpose_input(tgt_sents)
        ctx = dy.vecInput(self.args.hid_dim * 2)
        losses = []
        for i in range(1, maxLen):
            tgt_emb = dy.lookup_batch(self.tgt_lookup, tgt_wids[i - 1])
            x = dy.concatenate([tgt_emb, ctx])  # equation 74
            dec_state = dec_state.add_input(x)
            hid = dec_state.output()  # equaltion 74
            ctx, alpha_t = self.attention(encoding, hid, batch_size)
            read_out = dy.tanh(W_logit_cxt * ctx + W_logit_hid * hid + W_logit_input * tgt_emb + b_logit_readout)
            if self.args.dropout > 0:
                read_out = dy.dropout(read_out, self.args.dropout)
            y_t = dy.affine_transform([b_y, W_y, read_out])

            loss = dy.pickneglogsoftmax_batch(y_t, tgt_wids[i])
            if 0 in tgt_masks[i]:
                mask_expr = dy.inputVector(tgt_masks[i])
                # # print len(mask)
                mask_expr = dy.reshape(mask_expr, (1,), batch_size)
                loss = loss * mask_expr
            losses.append(loss)
        loss = dy.esum(losses)

        return dy.sum_batches(loss) / batch_size

    def gen_samples(self, src_seq, max_len=30, beam_size=5):
        encoding = self.encode([src_seq])

        W_init = dy.parameter(self.W_init)
        b_init = dy.parameter(self.b_init)

        W_logit_cxt = dy.parameter(self.W_logit_cxt)
        W_logit_input = dy.parameter(self.W_logit_input)
        W_logit_hid = dy.parameter(self.W_logit_hid)
        b_logit_readout = dy.parameter(self.b_logit_readout)

        softmax_w = dy.parameter(self.softmax_W)
        softmax_b = dy.parameter(self.softmax_b)

        state = self.dec_builder.initial_state(dy.tanh(dy.affine_transform(b_init, W_init, encoding[-1])))
        sample = [1]
        score = 0
        ctx = dy.vecInput(self.args.hid_dim * 2)

        hypotheses_pool = [(state, sample, score, ctx)]

        final_scores = []
        final_samples = []
        live = 1
        for i in range(max_len):
            cand_scores = []
            for k in range(live):
                y_t = dy.lookup(self.tgt_lookup, hypotheses_pool[k][1][-1])
                hypotheses_pool[k][0] = hypotheses_pool[k][0].add_input(dy.concatenate([y_t, hypotheses_pool[k][3]]))
                h_t = hypotheses_pool[k][0].output()
                hypotheses_pool[k][3], att_weights = self.attention(encoding, h_t, 1)
                read_out = dy.tanh(
                    W_logit_cxt * hypotheses_pool[k][3] + W_logit_hid * h_t + W_logit_input * y_t + b_logit_readout)
                prediction = dy.log_softmax(softmax_w * read_out + softmax_b).npvalue()
                cand_scores.append(hypotheses_pool[k][2] - prediction)

            cand_scores = np.concatenate(cand_scores).flatten()
            ranks = (cand_scores).argsort()[:(beam_size - len(final_scores))]

            cands_indices = ranks / self.tgt_vocab_size  # which k generated it
            cands_words = ranks % self.tgt_vocab_size  # which word is the next best for this k
            cands_scores = cand_scores[ranks]

            new_pool = []
            for idx, [bidx, widx] in enumerate(zip(cands_indices, cands_words)):
                prev_hyp = hypotheses_pool[bidx]
                new_hyp = (prev_hyp[0], prev_hyp[1] + [widx], prev_hyp[2] + cands_scores[idx], prev_hyp[3])
                if widx == 2:
                    final_scores.append(new_hyp[2])
                    final_samples.append(new_hyp[1])
                else:
                    new_pool.append(new_hyp)

            hypotheses_pool = new_pool

            live = beam_size - len(final_scores)

            if live == 0:
                break

        if live > 0:
            for idx in range(live):
                final_scores.append(hypotheses_pool[idx][2])
                final_samples.append(hypotheses_pool[idx][1])

        return final_scores, final_samples


def train(args):
    training_src = read_corpus(args.train_src)  # get vocabulary
    src_v = Vocab.from_corpus(training_src, args.src_vocab_size)
    src_train = get_data_id(src_v, training_src)
    src_vocab, src_id_to_words = src_v.w2i, src_v.i2w

    training_tgt = read_corpus(args.train_tgt)
    tgt_v = Vocab.from_corpus(training_tgt, args.tgt_vocab_size)
    tgt_train = get_data_id(tgt_v, training_tgt)
    tgt_vocab, tgt_id_to_words = tgt_v.w2i, tgt_v.i2w

    args.src_voc_size = len(src_vocab)
    args.tgt_voc_size = len(tgt_vocab)

    src_dev = get_data_id(src_v, read_corpus(args.dev_src))
    tgt_dev = get_data_id(tgt_v, read_corpus(args.dev_tgt))

    train_data = zip(src_train, tgt_train)
    dev_data = zip(src_dev, tgt_dev)

    model = EncoderDecoder(args, src_v, tgt_v, src_vocab, tgt_vocab)

    epochs = 30
    updates = 0
    valid_history = []
    bad_counter = 0
    total_loss = total_examples = total_length = 0
    start_time = time.time()
    eval_every = args.eval_every
    for epoch in range(epochs):
        for (src_batch, tgt_batch) in get_batches(train_data, args.batch_size):
            updates += 1
            batch_size = len(src_batch)

            if updates % eval_every == 0:

                bleu_score, translation = translate(model, dev_data, src_id_to_words, tgt_id_to_words)

                print "Epoch=%d, Updates=%d, Pairs_Porcessed=%d, BlEU score = %f " % (
                    epoch + 1, updates + 1, total_examples, bleu_score)

                if len(valid_history) == 0 or bleu_score > max(valid_history):
                    bad_counter = 0
                    model.save()
                    print "Bad counter back to 0 Model saved"
                else:
                    bad_counter += 1
                    print "Cautious, BLEU not decreasing, bad_counter " + str(bad_counter)
                    if bad_counter >= 10:
                        print("Early stop!")
                        exit(0)

                valid_history.append(bleu_score)

            src_encodings = model.encode(src_batch)
            decode_loss = model.decode_loss(src_encodings, tgt_batch)
            loss_value = decode_loss.value()

            total_loss += loss_value * batch_size
            total_examples += batch_size

            batch_length = sum([len(s) for s in tgt_batch])
            total_length += batch_length

            ppl = np.exp(loss_value * batch_size / batch_length)
            total_ppl = np.exp(total_loss / total_length)
            print "Epoch=%d, Updates=%d, Pairs_Porcessed=%d, Loss=%f, Avg. Loss=%f, PPL(for this batch)=%f, PPL(overall)=%f, Time taken=%d s" % \
                  (epoch + 1, updates + 1, total_examples, loss_value, total_loss / total_examples, ppl, total_ppl,
                   time.time() - start_time)
            decode_loss.backward()
            model.trainer.update()


def test(args):
    print "Beam size " + str(args.beam_size)
    training_src = read_corpus(args.train_src)  # get vocabulary
    src_v = Vocab.from_corpus(training_src, args.src_vocab_size)
    src_vocab, src_id_to_words = src_v.w2i, src_v.i2w

    training_tgt = read_corpus(args.train_tgt)
    tgt_v = Vocab.from_corpus(training_tgt, args.tgt_vocab_size)
    tgt_vocab, tgt_id_to_words = tgt_v.w2i, tgt_v.i2w

    args.src_voc_size = len(src_vocab)
    args.tgt_voc_size = len(tgt_vocab)

    src_test = get_data_id(src_v, read_corpus(args.test_src))
    tgt_test = get_data_id(tgt_v, read_corpus(args.test_tgt))
    test_data = zip(src_test, tgt_test)

    print "Test data line count total " + str(len(test_data))

    model = EncoderDecoder(args, src_v, tgt_v, src_vocab, tgt_vocab)
    model.load(args.model_name)

    bleu_score, translations = translate(model, test_data, src_id_to_words, tgt_id_to_words, args.beam_size)

    print  "BLEU on test data = " + str(bleu_score) + " " + str(args.beam_size)
    f_test = open("./model/" + args.model_name + "_test_translations_" + str(args.beam_size) + ".txt", "w")
    for hyp in translations:
        f_test.write(" ".join(hyp[1:-1]) + '\n')

    src_dev = get_data_id(src_v, read_corpus(args.dev_src))
    tgt_dev = get_data_id(tgt_v, read_corpus(args.dev_tgt))
    dev_data = zip(src_dev, tgt_dev)

    bleu_score, translations = translate(model, dev_data, src_id_to_words, tgt_id_to_words, args.beam_size)

    print  "BLEU on dev data = " + str(bleu_score) + " " + str(args.beam_size)
    f_dev = open("./model/" + args.model_name + "_dev_translations_" + str(args.beam_size) + ".txt", "w")
    for hyp in translations:
        f_dev.write(" ".join(hyp[1:-1]) + '\n')

    src_blind = get_data_id(src_v, read_corpus(args.blind_src))

    print "Blind data line count total " + str(len(src_blind))

    translations = translate_blind(model, src_blind, src_id_to_words, tgt_id_to_words, args.beam_size)

    f_blind = open("./model/" + args.model_name + "_blind_translations_" + str(args.beam_size) + ".txt", "w")
    for hyp in translations:
        f_blind.write(" ".join(hyp[1:-1]) + '\n')


def translate(model, data_pair, src_id_to_words, tgt_id_to_words, beam_size=5):
    translations = []
    references = []
    empty = True
    count = 0
    total = len(data_pair)
    for src_sent, tgt_sent in data_pair:
        count = count + 1

        scores, samples = model.gen_samples(src_sent, 200, beam_size)
        sample = samples[np.array(scores).argmin()]  # one of the best

        src = [src_id_to_words[i] for i in src_sent]
        tgt = [tgt_id_to_words[i] for i in tgt_sent]
        hyp = [tgt_id_to_words[i] for i in sample]

        if len(hyp) > 2:
            empty = False

        references.append([tgt])
        translations.append(hyp)

        print "\n" + str(count) + "/" + str(total)
        print  "Src sent: ", " ".join(src[1:-1])
        print  "Tgt sent: ", " ".join(tgt[1:-1])
        print  "Hypothesis: ", " ".join(hyp[1:-1])

    if empty:
        return 0.0, translations  # otherwise bleu will throw divided by zero error
    bleu_score = corpus_bleu(references, translations)
    return bleu_score, translations


def translate_blind(model, src_sents, src_id_to_words, tgt_id_to_words, beam_size=5):
    translations = []
    for src_sent in src_sents:
        scores, samples = model.gen_samples(src_sent, 200, beam_size)
        sample = samples[np.array(scores).argmin()]
        hyp = [tgt_id_to_words[i] for i in sample]
        translations.append(hyp)

    return translations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_src', type=str, default="./en-de/train.en-de.low.filt.de")
    parser.add_argument('--train_tgt', type=str, default="./en-de/train.en-de.low.filt.en")
    parser.add_argument('--dev_src', type=str, default="./en-de/valid.en-de.low.de")
    parser.add_argument('--dev_tgt', type=str, default="./en-de/valid.en-de.low.en")
    parser.add_argument('--test_src', type=str, default="./en-de/test.en-de.low.de")
    parser.add_argument('--test_tgt', type=str, default="./en-de/test.en-de.low.en")
    parser.add_argument('--blind_src', type=str, default="./en-de/blind.en-de.low.de")

    parser.add_argument('--src_vocab_size', type=int, default=30000)
    parser.add_argument('--tgt_vocab_size', type=int, default=20000)
    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--att_dim", type=int, default=256)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--eval_every', type=int, default=2500)
    parser.add_argument('--model_name', type=str, default="model")
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')

    parser.add_argument('--dynet-mem', default="6000,5000,1000", type=str)
    parser.add_argument('--random_seed', default=135109662, type=int)
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    if args.train:
        print "args.train True, invoking train()"
        train(args)
    else:
        print "args.train False, invoking test()"
        test(args)


class Hypothesis(object):
    def __init__(self, state, y, ctx_tm1, score):
        self.state = state
        self.y = y
        self.ctx_tm1 = ctx_tm1
        self.score = score

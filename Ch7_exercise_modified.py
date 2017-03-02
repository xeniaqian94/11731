import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import dynet as dy
import numpy as np
import argparse
from dynet import *
import copy

from nltk.translate.bleu_score import corpus_bleu
import time
# from utils import *
from util import *


def get_batches(sents_pair, batch_size):
    buckets = defaultdict(list)

    [buckets[len(pair[0])].append(pair) for pair in sents_pair]

    batches = []

    for len in buckets:
        bucket = buckets[len]
        np.random.shuffle(bucket)
        for i in range(int(np.ceil(len(bucket) * 1.0 / batch_size))):
            elements_count = np.min(batch_size, len(bucket) - batch_size * i)
            batches.append(([bucket[i * batch_size + j][0] for j in range(elements_count)],
                            [bucket[i * batch_size + j][1] for j in range(elements_count)]))

    np.random.shuffle(batches)
    for batch in batches:
        yield batch


class EncoderDecoder:
    def __init__(self, args, src_vocab, tgt_vocab, src_id_to_token, tgt_id_to_token):

        self.model = model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        self.args = args
        # self.src_token_to_id = args['src_token_to_id']
        self.src_vocab, self.src_token_to_id, self.src_id_to_token = src_vocab, src_vocab.w2i, src_vocab.i2w
        self.src_vocab_size = self.src_vocab.size()

        # self.tgt_token_to_id = args['tgt_token_to_id']
        self.tgt_vocab, self.tgt_token_to_id, self.tgt_id_to_token = tgt_vocab, tgt_vocab.w2i, tgt_vocab.i2w
        self.tgt_vocab_size = self.tgt_vocab.size()

        self.emb_size = args.emb_size
        self.hidden_dim = args.hid_dim
        self.att_dim = args.att_dim
        self.layers = args.layers
        self.beam_size = args.beam_size
        self.dropout = args.dropout
        self.concat_readout = args.concat_readout
        self.model_name = args.model_name

        self.src_emb = model.add_lookup_parameters((self.src_vocab_size, self.emb_size))
        self.tgt_emb = model.add_lookup_parameters((self.tgt_vocab_size, self.emb_size))
        self.enc_forward_rnn = GRUBuilder(1, self.emb_size, self.hidden_dim, model)
        self.enc_backward_rnn = GRUBuilder(1, self.emb_size, self.hidden_dim, model)

        self.W_init = model.add_parameters((self.hidden_dim, self.hidden_dim * 2))
        self.b_init = model.add_parameters((self.hidden_dim, 1))
        self.b_init.zero()
        # self.dec_rnn = GRUBuilder(config['layers'], self.emb_size, self.hidden_dim, model)
        self.dec_rnn = GRUBuilder(1, self.emb_size + self.hidden_dim * 2, self.hidden_dim, model)
        # read out parameters

        if args.concat_readout:
            # self.W_readout = model.add_parameters((self.tgt_voc_size, self.hidden_dim * 3))
            # self.b_readout = model.add_parameters((self.tgt_voc_size))
            self.W_readout = model.add_parameters((self.emb_size, self.hidden_dim * 3))
            self.b_readout = model.add_parameters((self.emb_size))
            self.b_readout.zero()
        else:
            self.W_logit_cxt = model.add_parameters((self.emb_size, self.hidden_dim * 2))
            self.W_logit_input = model.add_parameters((self.emb_size, self.emb_size))
            self.W_logit_hid = model.add_parameters((self.emb_size, self.hidden_dim))
            self.b_logit_readout = model.add_parameters((self.emb_size, 1))
            self.b_logit_readout.zero()

        # attention
        self.W_att_hidden = model.add_parameters((self.att_dim, self.hidden_dim))
        self.W_att_cxt = model.add_parameters((self.att_dim, self.hidden_dim * 2))
        self.V_att = model.add_parameters((1, self.att_dim))
        # self.b_att = model.add_parameters((1,))
        # self.b_att.zero()

        self.softmax_W = model.add_parameters((self.tgt_vocab_size, self.emb_size))
        self.softmax_b = model.add_parameters((self.tgt_vocab_size,))
        self.softmax_b.zero()
        self.SOS = 1
        self.EOS = 2

        # self.src_id_to_word = src_id_to_word
        # self.tgt_id_to_word = tgt_id_to_word

    def save(self):
        self.model.save("./model/" + self.model_name + "_params.bin")

    def load(self):
        self.model.load("./model/" + self.model_name + "_params.bin")

    def transpose_input(self, seq):
        max_len = max([len(sent) for sent in seq])
        seq_pad = []
        seq_mask = []
        for i in range(max_len):
            pad_temp = [sent[i] if i < len(sent) else 2 for sent in seq]
            mask_temp = [1 if i < len(sent) else 0 for sent in seq]
            seq_pad.append(pad_temp)
            seq_mask.append(mask_temp)
        return seq_pad, seq_mask

    def encode(self, src_seq):
        # src_seq is a batch with the same length
        dy.renew_cg()
        src_pad, src_mask = self.transpose_input(src_seq)
        wemb = [dy.lookup_batch(self.src_emb, wids) for wids in src_pad]  # (time_step, emb_size, batch_size)
        wemb_r = wemb[::-1]
        fwd_vectors = self.enc_forward_rnn.initial_state().transduce(wemb)
        bwd_vectors = self.enc_backward_rnn.initial_state().transduce(wemb_r)[::-1]

        seq_enc = [dy.concatenate([fwd_v, bwd_v]) for (fwd_v, bwd_v) in zip(fwd_vectors, bwd_vectors)]
        return seq_enc  # (time_step, hid_size*2, batch_size)

    def attention(self, encoding, hidden, batch_size):
        W_att_cxt = dy.parameter(self.W_att_cxt)
        W_att_hid = dy.parameter(self.W_att_hidden)
        V_att = dy.parameter(self.V_att)
        # b_att = dy.parameter(self.b_att)

        enc_seq = dy.concatenate_cols(encoding)  # (dim, time_step, batch_size)
        att_mlp = dy.tanh(dy.colwise_add(W_att_cxt * enc_seq, W_att_hid * hidden))

        # att_weights= dy.reshape(V_att * att_mlp + b_att, (len(encoding), ), batch_size)
        att_weights = dy.reshape(V_att * att_mlp, (len(encoding),), batch_size)

        att_weights = dy.softmax(att_weights)  # (time_step, batch_size)
        att_ctx = enc_seq * att_weights
        # print att_ctx.npvalue().shape
        return att_ctx, att_weights

    def decode_loss(self, encoding, tgt_seq):
        W_init_state = dy.parameter(self.W_init)
        b_init_state = dy.parameter(self.b_init)

        if self.concat_readout:
            W_readout = dy.parameter(self.W_readout)
            b_readout = dy.parameter(self.b_readout)
        else:
            W_logit_cxt = dy.parameter(self.W_logit_cxt)
            W_logit_input = dy.parameter(self.W_logit_input)
            W_logit_hid = dy.parameter(self.W_logit_hid)
            b_logit_readout = dy.parameter(self.b_logit_readout)

        softmax_w = dy.parameter(self.softmax_W)
        softmax_b = dy.parameter(self.softmax_b)

        # tgt sequence starts from <S>, ends at <\S>
        batch_size = len(tgt_seq)

        init_state = dy.tanh(dy.affine_transform([b_init_state, W_init_state, encoding[-1]]))
        # init_state = dy.tanh(W_init_state * encoding[-1] + b_init_state)
        dec_state = self.dec_rnn.initial_state([init_state])  # not sure

        tgt_pad, tgt_mask = self.transpose_input(tgt_seq)
        max_len = max([len(sent) for sent in tgt_seq])
        att_ctx = dy.vecInput(self.hidden_dim * 2)
        # shifted_tgt_emb = dy.concatenate(zero_emb + tgt_emb)
        # dec_states = self.dec_rnn.initial_state(enc_rep).transduce(shifted_tgt_emb)
        losses = []
        for i in range(max_len - 1):
            input_t = dy.lookup_batch(self.tgt_emb, tgt_pad[i])
            dec_state = dec_state.add_input(dy.concatenate([input_t, att_ctx]))
            ht = dec_state.output()
            att_ctx, att_weights = self.attention(encoding, ht, batch_size)
            if self.concat_readout:
                read_out = dy.tanh(dy.affine_transform([b_readout, W_readout, dy.concatenate([ht, att_ctx])]))
            else:
                read_out = dy.tanh(W_logit_cxt * att_ctx + W_logit_hid * ht + W_logit_input * input_t + b_logit_readout)
            if self.dropout > 0:
                read_out = dy.dropout(read_out, self.dropout)
            prediction = softmax_w * read_out + softmax_b

            loss = dy.pickneglogsoftmax_batch(prediction, tgt_pad[i + 1])

            if 0 in tgt_mask[i + 1]:
                mask_expr = dy.inputVector(tgt_mask[i + 1])
                mask_expr = dy.reshape(mask_expr, (1,), batch_size)
                loss = loss * mask_expr

            losses.append(loss)

        loss = dy.esum(losses)
        loss = dy.sum_batches(loss) / batch_size

        return loss

    def gen_samples(self, src_seq, max_len=30):
        encoding = self.encode([src_seq])
        beam_size = self.beam_size

        W_init_state = dy.parameter(self.W_init)
        b_init_state = dy.parameter(self.b_init)

        if self.concat_readout:
            W_readout = dy.parameter(self.W_readout)
            b_readout = dy.parameter(self.b_readout)
        else:
            W_logit_cxt = dy.parameter(self.W_logit_cxt)
            W_logit_input = dy.parameter(self.W_logit_input)
            W_logit_hid = dy.parameter(self.W_logit_hid)
            b_logit_readout = dy.parameter(self.b_logit_readout)

        softmax_w = dy.parameter(self.softmax_W)
        softmax_b = dy.parameter(self.softmax_b)

        live = 1
        dead = 0

        final_scores = []
        final_samples = []

        scores = np.zeros(live)
        dec_states = [
            self.dec_rnn.initial_state([dy.tanh(dy.affine_transform([b_init_state, W_init_state, encoding[-1]]))])]
        att_ctxs = [dy.vecInput(self.hidden_dim * 2)]
        samples = [[self.SOS]]

        for ii in range(max_len):
            cand_scores = []
            for k in range(live):
                y_t = dy.lookup(self.tgt_emb, samples[k][-1])
                dec_states[k] = dec_states[k].add_input(dy.concatenate([y_t, att_ctxs[k]]))
                h_t = dec_states[k].output()
                att_ctx, att_weights = self.attention(encoding, h_t, 1)
                att_ctxs[k] = att_ctx
                if self.concat_readout:
                    read_out = dy.tanh(dy.affine_transform([b_readout, W_readout, dy.concatenate([h_t, att_ctx])]))
                else:
                    read_out = dy.tanh(
                        W_logit_cxt * att_ctx + W_logit_hid * h_t + W_logit_input * y_t + b_logit_readout)
                prediction = dy.log_softmax(softmax_w * read_out + softmax_b).npvalue()
                cand_scores.append(scores[k] - prediction)

            cand_scores = np.concatenate(cand_scores).flatten()
            ranks = cand_scores.argsort()[:(beam_size - dead)]

            cands_indices = ranks / self.tgt_vocab_size
            cands_words = ranks % self.tgt_vocab_size
            cands_scores = cand_scores[ranks]

            new_scores = []
            new_dec_states = []
            new_att_ctxs = []
            new_samples = []
            for idx, [bidx, widx] in enumerate(zip(cands_indices, cands_words)):
                new_scores.append(copy.copy(cands_scores[idx]))
                new_dec_states.append(dec_states[bidx])
                new_att_ctxs.append(att_ctxs[bidx])
                new_samples.append(samples[bidx] + [widx])

            scores = []
            dec_states = []
            att_ctxs = []
            samples = []

            for idx, sample in enumerate(new_samples):
                if new_samples[idx][-1] == self.EOS:
                    dead += 1
                    final_samples.append(new_samples[idx])
                    final_scores.append(new_scores[idx])
                else:
                    dec_states.append(new_dec_states[idx])
                    att_ctxs.append(new_att_ctxs[idx])
                    samples.append(new_samples[idx])
                    scores.append(new_scores[idx])
            live = beam_size - dead

            if dead == beam_size:
                break

        if live > 0:
            for idx in range(live):
                final_scores.append(scores[idx])
                final_samples.append(samples[idx])

        return final_scores, final_samples

    def get_encdec_loss(self, src_seqs, tgt_seqs):
        # src_seq and tgt_seq are batches that are not padded to the same length
        encoding = self.encode(src_seqs)
        loss = self.decode_loss(encoding, tgt_seqs)
        return loss


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
    trainer = dy.AdamTrainer(model.model)

    epochs = 100
    updates = 0
    valid_history = []
    bad_counter = 0
    total_loss = total_examples = 0
    start_time = time.time()
    eval_every = args.eval_every
    for epoch in range(epochs):
        for (src_batch, tgt_batch) in get_batches(train_data, args.batch_size):
            updates += 1
            bs = len(src_batch)

            if updates % eval_every == 0:

                begin_time = time.time()
                bleu_score, translation = translate(model, dev_data, src_id_to_words, tgt_id_to_words)
                tt = time.time() - begin_time
                print  "BlEU score = %f. Time %d s elapsed. Avg decoding time per sentence %f s" % (
                    bleu_score, tt, tt * 1.0 / len(dev_data))

                if len(valid_history) == 0 or bleu_score > max(valid_history):
                    bad_counter = 0

                    model.save()
                    print "Model saved"
                else:
                    bad_counter += 1
                    if bad_counter >= args.patience:
                        print("Early stop!")
                        exit(0)

                valid_history.append(bleu_score)
            loss = model.get_encdec_loss(src_batch, tgt_batch)
            loss_value = loss.value()
            total_loss += loss_value * bs
            total_examples += bs

            ppl = np.exp(loss_value * bs / sum([len(s) for s in tgt_batch]))
            print  "Epoch=%d, Updates=%d, Loss=%f, Avg. Loss=%f, PPL=%f, Time taken=%d s" % \
                   (epoch + 1, updates + 1, loss_value, total_loss / total_examples, ppl,
                    time.time() - start_time)
            loss.backward()
            trainer.update()


def test(args, config):
    src_vocab, src_id_to_words, src_train = get_vocab(args.train_src, args.src_vocab_size)
    tgt_vocab, tgt_id_to_words, tgt_train = get_vocab(args.train_tgt, args.tgt_vocab_size)

    config["src_voc_size"] = len(src_vocab)
    config["tgt_voc_size"] = len(tgt_vocab)
    src_test = get_data(args.test_src, src_vocab)
    tgt_test = get_data(args.test_tgt, tgt_vocab)
    test_data = zip(src_test, tgt_test)

    nmt_model = EncoderDecoder(config)
    nmt_model.load()
    bleu_score, translations = translate(nmt_model, test_data, src_id_to_words, tgt_id_to_words)

    print  "BLEU on test data = ", bleu_score
    with open("./obj/" + args.model_name + "_test_hyps.txt", "w") as fout:
        for hyp in translations:
            fout.write(" ".join(hyp[1:-1]) + '\n')


def translate(model, data_pair, src_id_to_words, tgt_id_to_words):
    translations = []
    references = []
    empty = True
    count = 0
    for src_sent, tgt_sent in data_pair:
        count = count + 1
        if count > 100:
            break
        scores, samples = model.gen_samples(src_sent, 200)
        sample = samples[np.array(scores).argmin()]

        src = [src_id_to_words[i] for i in src_sent]
        tgt = [tgt_id_to_words[i] for i in tgt_sent]
        hyp = [tgt_id_to_words[i] for i in sample]

        if len(hyp) > 2:
            empty = False

        if len(hyp) == 2:
            print  "Empty translation!!!!!!!"
        references.append([tgt])
        translations.append(hyp)

        print  "Src sent: ", " ".join(src[1:-1])
        print  "Tgt sent: ", " ".join(tgt[1:-1])
        print  "Hypothesis: ", " ".join(hyp[1:-1])

    if empty:
        return 0.0, translations
    bleu_score = corpus_bleu(references, translations)
    return bleu_score, translations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--att_dim", type=int, default=256)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--train_tgt', type=str, default="./en-de/train.en-de.low.filt.en")
    parser.add_argument('--train_src', type=str, default="./en-de/train.en-de.low.filt.de")
    parser.add_argument('--dev_tgt', type=str, default="./en-de/valid.en-de.low.en")
    parser.add_argument('--dev_src', type=str, default="./en-de/valid.en-de.low.de")
    parser.add_argument('--test_tgt', type=str, default="./en-de/test.en-de.low.en")
    parser.add_argument('--test_src', type=str, default="./en-de/test.en-de.low.de")
    parser.add_argument('--src_vocab_size', type=int, default=30000)
    parser.add_argument('--tgt_vocab_size', type=int, default=20000)
    parser.add_argument('--load_from')
    parser.add_argument('--concat_readout', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=2500)
    parser.add_argument('--model_name', type=str, default="model")
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')

    parser.add_argument('--dynet-mem', default="6000,5000,1000", type=str)
    parser.add_argument('--random_seed', default=135109662, type=int)
    parser.add_argument('--for_loop_att', action="store_true", default=False)
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    if args.train:
        print "args.train True, invoking train()"
        train(args)
    else:
        print "args.train False, invoking test()"
        test(args, {})


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

        [w2i[key] for key, value in sorted_freqs[:top - 1] if value > 1]  # eliminate singleton
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


class Hypothesis(object):
    def __init__(self, state, y, ctx_tm1, score):
        self.state = state
        self.y = y
        self.ctx_tm1 = ctx_tm1
        self.score = score

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import _pickle as cPickle

import numpy as np
import tensorflow as tf

from server.tones_prediction.models import TransformerDecoder
from server.tones_prediction.tone_utils import clear_all_marks
from server.tones_prediction.data_load import basic_tokenizer, load_vocab


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TransformerDecoderPredictor():
    def __init__(self, vocab_path, ckpt_path, saved_args_path, beam_size=0):
        self.beam_size = beam_size
        with open(saved_args_path, 'rb') as f:
            self.saved_args = cPickle.load(f)
        self.word2idx, self.idx2word = load_vocab(vocab_path)

        self.ref = {}
        ind = 0
        for idx, token in self.idx2word.items():
            cleared = clear_all_marks(token)
            if cleared not in self.ref:
                self.ref[cleared] = ind
                ind += 1

        self.accent_vocab = {}
        for token, idx in self.word2idx.items():
            raw_token = clear_all_marks(token) if token[0] != "<" else token
            if raw_token not in self.accent_vocab:
                self.accent_vocab[raw_token] = [token]
            else:
                curr = self.accent_vocab[raw_token]
                if token not in curr:
                    curr.append(token)

        self.model = TransformerDecoder(False, self.saved_args)
        self.graph = self.model.graph
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.model.graph)
            self.sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)

    def predict(self, text):
        words = basic_tokenizer(text)[:self.saved_args.maxlen]
        pad_idx = self.word2idx["<pad>"]

        if self.beam_size:
            sos_idx = self.word2idx["<sos>"]
            init_state = np.full(
                shape=(self.saved_args.maxlen + 1), fill_value=pad_idx)
            init_state[0] = sos_idx
            with self.graph.as_default():
                init_probs = self.sess.run(
                    tf.nn.softmax(self.model.logits),
                    feed_dict={self.model.x: np.atleast_2d(init_state)})[0]
                paths, state_probs = self.beamsearch_transformer(
                    words, self.saved_args.maxlen, init_probs)
        else:
            x = np.asarray([self.ref[w] if w in self.ref
                            else self.ref["<unk>"] for w in words])
            x = np.atleast_2d(
                np.lib.pad(x, [0, self.saved_args.maxlen - len(x)],
                           'constant', constant_values=pad_idx))
            with self.graph.as_default():
                feed = {self.model.x: x}
                paths = [self.sess.run(self.model.preds, feed_dict=feed)]
                paths[0][len(words):] = pad_idx

        results = []
        for path in paths:
            tokens = []
            for idx, token in enumerate(path):
                if token == pad_idx:
                    break
                if token != self.word2idx["<unk>"] and token != -1 and \
                        token < len(self.idx2word):
                    tokens.append(self.idx2word[token])
                else:
                    tokens.append(words[idx])

            results.append(tokens)
        return results

    def predict_json(self, inputs):
        results = self.predict(inputs['sentence'])

        result_texts = []
        for tokens in results:
            result_texts.append(' '.join(tokens))
        return {
            'words': [result_texts]
        }

    def find_candidates(self, probs, word):
        if word not in self.accent_vocab:
            return np.array([self.word2idx["<unk>"]]), np.array([1])
        else:
            possible_candidates = np.asarray(
                [self.word2idx[x] for x in self.accent_vocab[word]])
            possible_candidates_probs = probs[possible_candidates]
            top_k = np.argsort(-possible_candidates_probs)[:self.beam_size]
            possible_candidates = possible_candidates[top_k]
            return possible_candidates, probs[possible_candidates]

    def beamsearch_transformer(self, sent, maxlen, initial_probs):
        stop_token = self.word2idx["<pad>"]
        num_tokens = len(sent)
        state_probs = np.zeros(self.beam_size, dtype=np.float32)
        all_candidates = np.zeros(
            [self.beam_size, self.beam_size], dtype=np.int32)
        all_candidates_probs = np.zeros(
            [self.beam_size, self.beam_size], dtype=np.float32)
        paths = np.zeros([self.beam_size, num_tokens], dtype=np.int32)
        states = np.zeros([self.beam_size, maxlen + 1], dtype=np.int32)
        sum_probs = 0.0

        # initializing first graph level
        top_k, top_k_probs = self.find_candidates(initial_probs, sent[0])
        for i in range(self.beam_size):
            state_probs[i] = initial_probs[top_k[i]] if i < len(top_k) else 0
            sum_probs += state_probs[i]
            paths[i, 0] = top_k[i] if i < len(top_k) else stop_token

        for i in range(self.beam_size):
            state_probs[i] /= sum_probs
            for level in range(1, num_tokens):
                paths[i, level] = stop_token
            states[i, 1:num_tokens + 1] = paths[i].copy()
            states[i, 0] = self.word2idx["<sos>"]
        top_k = list(top_k)

        # for each graph level
        for level in range(1, num_tokens):
            for s_idx in range(self.beam_size):
                if s_idx >= len(top_k):
                    top_k.append(top_k[0])
                feed = states[s_idx].copy()
                initial_probs = self.sess.run(
                    self.model.logits, feed_dict={
                        self.model.x: np.atleast_2d(feed)})[level]
                initial_probs = self.sess.run(tf.nn.softmax(initial_probs))

                top_prob_indices, _ = self.find_candidates(
                    initial_probs, sent[level])

                for k in range(len(top_prob_indices)):
                    sum_probs += state_probs[s_idx] * \
                        initial_probs[top_prob_indices[k]]
                for k in range(self.beam_size):
                    all_candidates[s_idx, k] = top_prob_indices[k] \
                        if k < len(top_prob_indices) else -1
                    all_candidates_probs[s_idx, k] = state_probs[s_idx] * \
                        initial_probs[top_prob_indices[k]] / sum_probs \
                        if k < len(top_prob_indices) else 0

            # find top k among all_candidates_probs
            top_candidates = np.argsort(
                -all_candidates_probs.reshape(-1))[:self.beam_size]
            tmp_path = np.zeros([self.beam_size, num_tokens], dtype=np.int32)
            for s_idx in range(self.beam_size):
                top_k[s_idx] = all_candidates.flat[top_candidates[s_idx]]
                state_probs[s_idx] = all_candidates_probs.flat[
                    top_candidates[s_idx]]
                state_idx = int(top_candidates[s_idx] / self.beam_size)
                # update path
                for c in range(num_tokens):
                    if c < level:
                        tmp_path[s_idx, c] = paths[state_idx, c]
                    else:
                        tmp_path[s_idx, c] = stop_token
                tmp_path[s_idx, level] = top_k[s_idx]
                states[s_idx, 1:num_tokens + 1] = tmp_path[s_idx]

                sum_probs += state_probs[s_idx]

            for s_idx in range(self.beam_size):
                state_probs[s_idx] /= sum_probs
            paths = tmp_path

        return paths, state_probs

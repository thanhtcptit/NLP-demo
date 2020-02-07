import os
import time

import numpy as np
import tensorflow as tf

from nsds.common import Params
from server.utils import load_model, clear_all_marks, basic_tokenizer, \
    load_vocab


class TransformerDecoderPredictor():
    def __init__(self, config_path, model_path, beam_size=0,
                 in_graph_beam_search=True):
        params = Params.from_file(config_path)
        self.data_params = params['data']
        self.model_params = params['model']

        self.max_length = self.data_params['max_length']
        self.beam_size = beam_size
        self.in_graph_beam_search = in_graph_beam_search
        self.word2idx, self.idx2word = load_vocab(
            self.data_params['vocab_path'])

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

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(config=sess_config)

        load_model(model_path, session=self.sess)
        self.x = tf.get_default_graph().get_tensor_by_name(
            'Placeholder:0')
        try:
            self.logits = tf.get_default_graph().get_tensor_by_name(
                'decoder/MatMul:0')
        except KeyError:
            self.logits = tf.get_default_graph().get_tensor_by_name(
                'MatMul:0')

        self.prediction = tf.get_default_graph().get_tensor_by_name(
            'ToInt32:0')
        self.probs = tf.nn.softmax(self.logits)

    def predict(self, text):
        words = basic_tokenizer(text)[:self.max_length]
        pad_idx = self.word2idx["<pad>"]

        if self.beam_size:
            if self.in_graph_beam_search:
                x = np.asarray([self.ref[w] if w in self.ref
                               else self.ref["<unk>"] for w in words])
                x = np.atleast_2d(
                    np.lib.pad(x, [0, self.max_length - len(x) + 1],
                               'constant', constant_values=pad_idx))
                res = self.sess.run(tf.nn.ctc_beam_search_decoder(
                    tf.expand_dims(self.logits, 1), [self.max_length], 3,
                    merge_repeated=False),
                    feed_dict={self.x: x})
                paths = [res[0][0].values[:len(words)]]
            else:
                sos_idx = self.word2idx["<sos>"]
                init_state = np.full(
                    shape=(self.max_length + 1), fill_value=pad_idx)
                init_state[0] = sos_idx
                init_probs = self.sess.run(
                    self.probs,
                    feed_dict={self.x: np.atleast_2d(init_state)})[0]
                paths, state_probs = self.beam_search(
                    words, self.max_length, init_probs)
        else:
            x = np.asarray([self.ref[w] if w in self.ref
                            else self.ref["<unk>"] for w in words])
            x = np.atleast_2d(
                np.lib.pad(x, [0, self.max_length - len(x)],
                           'constant', constant_values=pad_idx))
            paths = [self.sess.run(self.prediction, feed_dict={self.x: x})]
            paths[0][len(words):] = pad_idx

        results = []
        for path in paths:
            tokens = []
            for idx, token in enumerate(path):
                if token == pad_idx:
                    break
                if token != self.word2idx["<unk>"] and token != -1 and \
                        token < len(self.idx2word) and \
                        clear_all_marks(self.idx2word[token]) == words[idx]:
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

    def beam_search(self, sent, max_length, initial_probs):
        stop_token = self.word2idx["<pad>"]
        num_tokens = len(sent)
        state_probs = np.zeros(self.beam_size, dtype=np.float32)
        all_candidates = np.zeros(
            [self.beam_size, self.beam_size], dtype=np.int32)
        all_candidates_probs = np.zeros(
            [self.beam_size, self.beam_size], dtype=np.float32)
        paths = np.zeros([self.beam_size, num_tokens], dtype=np.int32)
        states = np.zeros([self.beam_size, max_length + 1], dtype=np.int32)
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
                    self.probs, feed_dict={
                        self.x: np.atleast_2d(feed)})[level]

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


if __name__ == '__main__':
    s = time.time()
    cfg_path = 'resources/transformer_decoder/config/transformer_decoder.json'
    model_path = 'resources/transformer_decoder/model_epoch_06_gs_60792.pb'
    model = TransformerDecoderPredictor(cfg_path, model_path, beam_size=0)
    print(model.predict('Mot con vit xoe ra hai cai canh'))
    print(time.time() - s)

import os
import time

import numpy as np
import tensorflow as tf

from nsds.common import Params

from server.data.tokenization import Vocab
from server.utils import load_model, clear_all_marks


class TransformerDecoderPredictor:
    def __init__(self, config_path, model_path, beam_size=0,
                 in_graph_beam_search=True):
        params = Params.from_file(config_path)
        self.data_params = params['data']

        self.max_length = self.data_params['max_length']
        self.target_vocab_size = self.data_params['target_vocab_size']

        self.beam_size = beam_size
        self.in_graph_beam_search = in_graph_beam_search
        self.vocab = Vocab(self.data_params['target_vocab_path'])

        self.accent_vocab = {}
        for token, idx in self.vocab.word2idx.items():
            if not token:
                continue
            raw_token = clear_all_marks(token) if token[0] != "<" else token
            if raw_token not in self.accent_vocab:
                self.accent_vocab[raw_token] = [token]
            else:
                if token not in self.accent_vocab[raw_token]:
                    self.accent_vocab[raw_token].append(token)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(config=sess_config)

        load_model(model_path, session=self.sess)
        self.x = tf.get_default_graph().get_tensor_by_name(
            'input_ids:0')
        self.logits = tf.get_default_graph().get_tensor_by_name(
            'logits:0')
        self.prediction = tf.get_default_graph().get_tensor_by_name(
            'predictions:0')
        self.probs = tf.get_default_graph().get_tensor_by_name(
            'probabilities:0')

    def predict(self, text):
        txt = text.lower()
        words = [self.vocab.get_sos_token()] + \
            self.vocab.tokenize(text)[:self.max_length - 2] + \
            [self.vocab.get_eos_token()]
        sos_idx = self.vocab.get_sos_token_id()
        eos_idx = self.vocab.get_eos_token_id()
        pad_idx = self.vocab.get_pad_token_id()

        if self.beam_size:
            if self.in_graph_beam_search:
                x = np.asarray(self.vocab.tokens_to_ids(words), dtype=np.int32)
                x = np.atleast_2d(
                    np.lib.pad(x, [0, self.max_length - len(x)],
                               'constant', constant_values=pad_idx))
                res = self.sess.run(tf.nn.ctc_beam_search_decoder(
                    tf.expand_dims(self.logits, 1), [self.max_length],
                    beam_width=self.beam_size, merge_repeated=False),
                    feed_dict={self.x: x})
                paths = [res[0][0].values[:len(words)]]
            else:
                sos_idx = self.vocab.get_sos_token()
                init_state = np.full(
                    shape=(self.max_length), fill_value=pad_idx)
                init_state[0] = sos_idx
                init_probs = self.sess.run(
                    self.probs,
                    feed_dict={self.x: np.atleast_2d(init_state)})[0]
                paths, state_probs = self.beam_search(
                    words, self.max_length, init_probs)
        else:
            x = np.asarray(self.vocab.tokens_to_ids(words), dtype=np.int32)
            x = np.atleast_2d(
                np.lib.pad(x, [0, self.max_length - len(x)],
                           'constant', constant_values=pad_idx))
            paths = self.sess.run(self.prediction, feed_dict={self.x: x})

        results = []
        for path in paths:
            tokens = []
            for idx, token in enumerate(path):
                if token == sos_idx:
                    continue
                if token in [eos_idx, pad_idx]:
                    break
                if token != self.vocab.get_unk_token_id() and \
                        clear_all_marks(
                            self.vocab.idx2word[token]) == words[idx]:
                    tokens.append(self.vocab.idx2word[token])
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
            return np.array([self.vocab.get_unk_token()]), np.array([1])
        else:
            possible_candidates = np.asarray(
                [self.vocab.tokens_to_ids(self.accent_vocab[word])])
            possible_candidates_probs = probs[possible_candidates]
            top_k = np.argsort(-possible_candidates_probs)[:self.beam_size]
            possible_candidates = possible_candidates[top_k]
            return possible_candidates, probs[possible_candidates]

    def beam_search(self, sent, max_length, initial_probs):
        stop_token = self.vocab.get_pad_token()
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
            states[i, 0] = self.vocab.get_sos_token()
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
    cfg_path = "resources/transformer_decoder_fim/config.json"
    model_path = "resources/transformer_decoder_fim/model.pb"
    model = TransformerDecoderPredictor(cfg_path, model_path, beam_size=0)

    from server.utils import remove_accent
    text = 'cô gái đến từ hôm qua'
    inp = remove_accent(text)
    print(inp)
    print(model.predict(inp))
    print(time.time() - s)
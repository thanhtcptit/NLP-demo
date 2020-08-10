import os

import tensorflow as tf

from server.utils import load_model
from server.data import tokenization


class MaskedLMPredictor:
    def __init__(self, model_path, vocab_path, max_seq_length,
                 uncased=False):
        self.max_seq_length = max_seq_length
        self.uncased = uncased

        self.sess = tf.Session()
        load_model(model_path, session=self.sess)

        self.input_ids = tf.get_default_graph().get_tensor_by_name(
            'input_ids:0')
        self.input_mask = tf.get_default_graph().get_tensor_by_name(
            'input_mask:0')
        self.segment_ids = tf.get_default_graph().get_tensor_by_name(
            'segment_ids:0')
        self.masked_lm_position = tf.get_default_graph().get_tensor_by_name(
            'masked_lm_position:0')
        self.top_k_log_probs = tf.get_default_graph().get_tensor_by_name(
            'top_k:0')
        self.top_k_preds = tf.get_default_graph().get_tensor_by_name(
            'top_k:1')

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=False)

    def predict(self, text):
        if self.uncased:
            text = text.lower()
        inp_ids = self.tokenizer.tokenize(text)
        inp_ids = ['[CLS]'] + inp_ids + ['[MASK]'] + ['[SEP]']
        print(inp_ids)

        masked_lm_pos = [inp_ids.index('[MASK]')]
        inp_ids = self.tokenizer.convert_tokens_to_ids(inp_ids)
        inp_mask = [1] * len(inp_ids)
        seg_ids = [0] * len(inp_ids)

        while len(inp_ids) < self.max_seq_length:
            inp_ids.append(0)
            inp_mask.append(0)
            seg_ids.append(0)

        top_k_p, top_k_id = self.sess.run(
            [self.top_k_log_probs, self.top_k_preds],
            feed_dict={
                self.input_ids: [inp_ids],
                self.input_mask: [inp_mask],
                self.segment_ids: [seg_ids],
                self.masked_lm_position: [masked_lm_pos]
            }
        )
        top_k_p = top_k_p[0].tolist()
        top_k_id = top_k_id[0].tolist()
        top_k_words = self.tokenizer.convert_ids_to_tokens(top_k_id)
        suggest_terms = []
        for w in top_k_words:
            if w[0] != '#':
                w = ' ' + w
            else:
                w = w[2:]
            suggest_terms.append(w)
        return top_k_p, suggest_terms

    def predict_json(self, inputs):
        top_k_p, top_k_words = self.predict(inputs['sentence'])
        return {
            "probabilities": [top_k_p],
            "words": [top_k_words]
        }


if __name__ == '__main__':
    import time
    max_seq_length = 32
    model_dir = 'train_logs/multi_cased_fim_finetune'
    vocab_path = os.path.join(model_dir, 'vocab.txt')
    model_path = os.path.join(model_dir, 'predict_model', 'model.pb')

    predictor = MaskedLMPredictor(model_path, vocab_path, max_seq_length)
    start = time.time()
    print(predictor.predict("cho max anh"))
    print(time.time() - start)

    start = time.time()
    print(predictor.predict("trang"))
    print(time.time() - start)

import os
import re
import codecs


_WORD_SPLIT = re.compile("([.,!?\"':;)(])")


def read(file_path):
    with open(file_path, encoding='utf-8') as f:
        return f.read().split('\n')


def load_vocab(path):
    vocab = [line.split()[0] for line in codecs.open(
        path, 'r', 'utf-8').read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w.lower() for w in words if w != '' and w != ' ']

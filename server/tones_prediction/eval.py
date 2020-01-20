import os

from tqdm import tqdm

from server.tones_prediction.tone_utils import remove_accent
from server.tones_prediction.data_load import basic_tokenizer
from server.tones_prediction.transfomer_decoder import \
    TransformerDecoderPredictor


vocab_path = "models/transformer-decoder/vocab.txt"
ckpt_path = "models/transformer-decoder/"
saved_args_path = "models/transformer-decoder/args.pkl"
beam_size = 3

model = TransformerDecoderPredictor(vocab_path, ckpt_path, saved_args_path,
                                    beam_size)

data_path = 'data/myclip_titles.txt'
error_log = 'data/td_err.log'

total_words = 0
correct_words = 0

with open(error_log, 'w') as ef:
    with open(data_path) as f:
        for line in tqdm(f):
            line = line.strip().lower()
            tokens = basic_tokenizer(line)
            non_accent_line = remove_accent(line)
            prediction = model.predict(non_accent_line)[0]

            flag = False
            for pred, label in zip(prediction, tokens):
                if pred == label:
                    correct_words += 1
                else:
                    flag = True
                total_words += 1

            if flag:
                ef.write(' '.join(prediction) + ' | ' + line + '\n')

print(f'Accuracy: {correct_words / total_words}')

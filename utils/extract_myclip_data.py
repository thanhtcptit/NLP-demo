import os
import re

import pandas as pd
from tqdm import tqdm


non_alphabet_filter = re.compile(r'["()/\\[\]<>|*…:%@#$^“”_&★►●\-=+–\']')
end_sentence_delimiters = "([.,!?\"':;)(])"


def get_clean_sentences(description):
    if not isinstance(description, str):
        return []
    clean_description = non_alphabet_filter.sub(' ', description).replace(
        '...', '')
    clean_description = re.sub('\s+', ' ', clean_description)
    sentences = re.split(end_sentence_delimiters, clean_description)
    return sentences


data = pd.read_csv('data/video.csv', usecols=['name', 'description'])

with open('data/myclip_titles.txt', 'w') as f:
    for i, row in tqdm(data.iterrows()):
        sentences = get_clean_sentences(row['name'])
        #   + get_clean_sentences(row['description'])
        for s in sentences:
            if len(s) >= 5:
                f.write(s.strip() + '.\n')

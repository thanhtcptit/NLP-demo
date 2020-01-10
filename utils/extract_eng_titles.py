import pandas as pd


imdb_data = pd.read_csv('data/title.akas.tsv', usecols=['title', 'region'],
                        delimiter='\t')
imdb_data = imdb_data.loc[imdb_data['region'] == 'US']

with open('data/imdb_titles_eng.txt', 'w') as f:
    for i, row in imdb_data.iterrows():
        f.write(row['title'] + '\n')

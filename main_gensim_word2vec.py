#coding utf-8
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim import models
from gensim.models import word2vec
from gensim.models import KeyedVectors

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import pandas as pd

import load_datasets
import text_tokenize

INPUT_TRAIN = './datasets/poems.csv'
INPUT_TEST = './datasets/poems.csv'


def predict(model):
    result = model.most_similar(positive="東京")

    for pair in result:
        print(pair[0], pair[1])

 
def load_model(filepath):
    model = word2vec.Word2Vec.load(filepath)
    #model = KeyedVectors.load_word2vec_format("./entity_vector/entity_vector.model.bin", binary=True)

    return model

if __name__ == '__main__':
    dataset = load_datasets.Datasets(INPUT_TRAIN, INPUT_TEST, header=0)
    df, test_df = dataset.load_csv()

    # [['word1', 'word2'], ['word3', 'word4']...]
    # [4:]でname 冒頭の'nnn.'を削除
    tokenized_text_list = [text_tokenize.stems(d[4:]) for i, d in df['name'].items()]

    #print(tokenized_text_list)
    model = word2vec.Word2Vec(tokenized_text_list, size=150, min_count=2, window=15, iter=15)

    model.save("poem.model")

    # predict
    model = load_model('poem.model')
    predict(model)

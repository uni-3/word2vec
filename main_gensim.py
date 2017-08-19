#coding utf-8
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim import models
from gensim.models import word2vec
from gensim.models import KeyedVectors
#from gensim.models.doc2vec import LabeledSentence

import pandas as pd

import load_datasets
import text_tokenize

INPUT_TRAIN = './datasets/train.tsv'
INPUT_TEST = './datasets/test.tsv'

# 参考記事： http://qiita.com/okappy/items/32a7ba7eddf8203c9fa1
class LabeledListSentence(object):
    def __init__(self, words_list, labels):
        self.words_list = words_list
        self.labels = labels

    def __iter__(self):
        for i, words in enumerate(self.words_list):
            yield models.doc2vec.LabeledSentence(words, ['%s' % self.labels[i]])

def load_model(filepath):
    model = word2vec.Word2Vec.load(filepath)
    #model = KeyedVectors.load_word2vec_format("./entity_vector/entity_vector.model.bin", binary=True)

    return model

if __name__ == '__main__':
    dataset = load_datasets.Datasets()
    df, test_df = dataset.load_tsv()
    tokenized_text_list = text_tokenize.csv_tokenaze(df)
    labels = [d[1] for i, d in df.iterrows()]
    #print(tokenized_text_list)
    #print(labels)
    sentences = LabeledListSentence(tokenized_text_list, labels)

    #model = KeyedVectors.load_word2vec_format('./entity_vector/entity_vector.model.bin')

    model = models.Doc2Vec(alpha=0.025, min_count=5,
                           size=100, iter=20, workers=2)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(tokenized_text_list), epochs=1000)

    model.save("test.model")
    model.save_word2vec_format("test_word2vec.model")

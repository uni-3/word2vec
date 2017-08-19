#coding utf-8

from gensim import models

import MeCab

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


import plotly.offline as offline
offline.init_notebook_mode()
import plotly.graph_objs as go
import plotly.plotly as py

import load_datasets
import text_tokenize

INPUT_TEST = './datasets/test.tsv'

def t_sne(model, limit=100, skip=0):
    vocab = model.__dict__['index2word']
    emb_tuple = tuple([model[v] for v in vocab]) # 各ベクトルのtuple
    #print("tuple", emb_tuple)

    X = np.vstack(emb_tuple)

    model_tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    tsne = model_tsne.fit_transform(X)

    labels = vocab
    x = model_tsne.embedding_[:, 0]
    y = model_tsne.embedding_[:, 1]

    trace = go.Scatter(x=np.array(tsne[:,0]), y=np.array(tsne[:,1])
                       , mode="marker+text"
                       , marker=dict(
                                size=4, sizemode='diameter', color='rgba(255, 0, 255, 0.5)'
                        )
                       , text=labels
                       )
    layout = go.Layout(
        title="テスト"
        , xaxis=dict(title='distance')
        , yaxis=dict(title='distance')
    )
    data = [trace]
    fig = dict(data=data, layout=layout)
    #offline.plot(fig, filename="test.html", image="png")
    #offline.iplot(fig, filename="test", image="png")
    offline.plot(fig, image="png")


if __name__ == '__main__':
    dataset = load_datasets.Datasets()
    train_df, df = dataset.load_tsv()
    tokenized_text_list = text_tokenize.csv_tokenaze(df)
    #model = models.KeyedVectors.load('./test.model', binary=True)
    model = models.KeyedVectors.load_word2vec_format('./test_word2vec.model')

    t_sne(model, limit=100, skip=0)

    """
    predicts = []
    for i, tokenized_text in enumerate(tokenized_text_list):
        for j, tokenized in enumerate(tokenized_text):
            try:
                pre= model.most_similar(positive=tokenized)
            except KeyError:
                print('key error')
                pass
            row = "id ," + str(i) + "," + pre[0][0]
            predicts.append(row)
            print("id ,", i, "," , pre[0])

    df_pre = pd.DataFrame(predicts)
    df_pre.to_csv("predicts.csv")
    """

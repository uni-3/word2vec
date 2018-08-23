from gensim import models
from gensim.models import word2vec
from gensim.models import KeyedVectors
#from gensim.models.doc2vec import LabeledSentence

import MeCab

def _split_to_words(text, to_stem=False, ex=['名詞', '動詞', '形容詞', '接頭詞', '連体詞', '助動詞', '感動詞']):
    """
    入力: 'すべて自分のほうへ'
    出力: tuple(['すべて', '自分', 'の', 'ほう', 'へ'])
    """
    tagger = MeCab.Tagger('mecabrc')  # 別のTaggerを使ってもいい
    mecab_result = tagger.parse(text)
    info_of_words = mecab_result.split('\n')
    words = []
    for info in info_of_words:
        # macabで分けると、文の最後に’’が、その手前に'EOS'が来る
        if info == 'EOS' or info == '':
            break
            # info => 'な\t助詞,終助詞,*,*,*,*,な,ナ,ナ'
        info_elems = info.split(',')
        # 特定の品詞を抽出
        hinshi =  info_elems[0].split('\t')[1   ]
        if hinshi not in ex:
            continue
        #print('word', info_elems[0])

        if to_stem:
            # 語幹に変換
            word = info_elems[6]
            #words.append(info_elems[6])
            #continue
        # 6番目に、無活用系の単語が入る。もし6番目が'*'だったら0番目を入れる
        if info_elems[6] == '*':
            # info_elems[0] => 'ヴァンロッサム\t名詞'
            word = info_elems[0][:-3]
            #words.append(info_elems[0][:-3])
            #continue
        # 英語表記ゆれ対応

        word = word.lower()
        words.append(word)
        print('word', word)
        #words.append(info_elems[0][:-3])
    return words


def words(text):
    words = _split_to_words(text=text, to_stem=False)
    return words


def stems(text):
    stems = _split_to_words(text=text, to_stem=True)
    return stems

def tokenize(text):
    wakati = MeCab.Tagger('-O wakati')
    #wakati = MeCab.Tagger('-O chasen')
    return wakati.parse(text)

def csv_tokenaze(df):
    #tokenized_text_list = [tokenize(texts) for texts in df[0]]
    tokenized_text_list = [stems(d[0]) for i, d in df.iterrows()]
    return tokenized_text_list

if __name__ == '__main__':
    tokenized_text_list = csv_tokenaze(df)
    labels = [d[1] for i, d in df.iterrows()]
    #print(tokenized_text_list)
    #print(labels)
    sentences = LabeledListSentence(tokenized_text_list, labels)

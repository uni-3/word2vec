#coding utf-8
import pandas as pd

INPUT_TRAIN = './datasets/train.tsv'
INPUT_TEST = './datasets/test.tsv'

class Datasets(object):
    def __init__(self, input_train=INPUT_TRAIN, input_test=INPUT_TEST, header=-1):
        self.input_train = input_train
        self.input_test = input_test
        self.header = header

    def load_tsv(self):
        train_df = pd.read_csv(self.input_train, header=self.header, sep='\t')
        test_df = pd.read_csv(self.input_test, header=self.header, sep='\t')
        return train_df, test_df

    def load_csv(self):
        train_df = pd.read_csv(self.input_train, header=self.header)
        test_df = pd.read_csv(self.input_test, header=self.header)
        return train_df, test_df


if __name__ == '__main__':
    ds = Datasets(INPUT_TRAIN, INPUT_TEST)
    train_df, test_df = ds.load_tsv()
    print(train_df)

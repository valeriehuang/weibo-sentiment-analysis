import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import fasttext
from fasttext import load_model
import os
import numpy as np


def get_file_names(input_directory):
    result_list = []
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):
            result_list.append(file_name)
    return result_list


# id,label,keyword,date,content,time,forward,comment,like
def gather_data(input_directory):
    labeled_file_list = get_file_names(input_directory)
    all_data = pd.DataFrame()
    for file in labeled_file_list:
        path = input_directory + "/" + file
        data = pd.read_csv(path)
        all_data = pd.concat([all_data, data], ignore_index=True)
    # all_data.to_csv('all_train_data.csv', index=False)
    return all_data


def train(df):
    # df = pd.read_csv(input_file)
    df = df[['label', 'content', 'time', 'forward', 'comment', 'like']].copy()
    df['label'] = df['label'].apply(lambda x: str(x) + ' ')
    df['content'] = df['content'].apply(lambda x: ' ' + str(x))
    # train.txt only need content and __label__
    df = df[['label', 'content']]
    df.to_csv('train.txt', header=None, index=None)

    classifier = fasttext.train_supervised('train.txt', label="__label__", dim=200, lr=0.2,
                                           epoch=25, wordNgrams=2, loss='ova')
    classifier.save_model('my_senti_model.bin')


def test(test_input, test_output):

    df = pd.read_csv(test_input)
    # df.columns = ['label', 'content', 'time', 'forward', 'comment', 'like']

    content = df['content'].tolist()
    classifier = load_model('my_senti_model.bin')
    result = classifier.predict(content, k=3, threshold=0)

    label_result = np.array(result[0])
    score_result = np.array(result[1]).round(5)
    df['predict'] = label_result[:, 0]
    df['__label__positive'] = ""
    df['__label__negative'] = ""
    df['__label__neutral'] = ""
    df['predict_score'] = score_result[:, 0]

    for i in range(len(df)):
        j = 0
        for label in label_result[i]:
            df.loc[i, label] = score_result[i][j]
            j += 1
        i += 1

    df.to_csv(test_output, index=False)


def test_all(input_directory):
    file_list = get_file_names(input_directory)
    for file in file_list:
        input_path = input_directory + '/' + file
        output_path = 'test_result/' + file
        test(input_path, output_path)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Invalid argument")
        print("Usage:\ntrain_test.py [train_directory] [test_directory]")
        exit(0)
    # train model
    train_directory = sys.argv[1]
    train_df = gather_data(train_directory)
    train(train_df)

    # predict on test data
    test_directory = sys.argv[2]
    test_df = gather_data(test_directory)
    test_all(test_directory)

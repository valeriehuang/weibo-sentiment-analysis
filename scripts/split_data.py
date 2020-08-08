import sys
import os
import pandas as pd


def get_file_names(input_directory):
    result_list = []
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):
            result_list.append(file_name)
    return result_list


# only split the data for the labeled data
def split_train_test(input_directory):
    file_list = get_file_names(input_directory)
    for file in file_list:
        path = input_directory + '/' + file
        df = pd.read_csv(path)
        train = df.sample(frac=0.8, random_state=200)
        test = df.drop(train.index)
        train_file_name = 'data_train/' + file
        train.to_csv(train_file_name, index=False)
        test_file_name = 'data_test/' + file
        test.to_csv(test_file_name, index=False)


# mark the rest of data to test data
# 2020-01-10 to 2020-01-26
# 2020-03-15 to 2020-03-26
def mark_test_data(input_directory):
    file_list = get_file_names(input_directory)
    for file in file_list:
        path = input_directory + '/' + file
        df = pd.read_csv(path)
        df = df.drop_duplicates(subset='content', keep='first', inplace=False)
        df['label'] = -1
        out_file_name = 'data_test/' + file
        df.to_csv(out_file_name, index=False)


if __name__ == '__main__':
    # split labeled data to train and test
    labeled_input_directory = '../labeled_data'
    split_train_test(labeled_input_directory)
    # mark unlabeled data to test data
    unlabeled_input_directory = '../unlabeled_data'
    mark_test_data(unlabeled_input_directory)

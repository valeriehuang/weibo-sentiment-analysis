import pandas as pd
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

import gensim
from gensim import corpora
import sys
import os

# data precesssing
stop_words = stopwords.words('chinese')
stop_words.extend([u'新冠', u'肺炎', u'疫情', u'冠状病毒', u'新型'])
stop = set(stop_words)
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num,5)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def get_topic(raw_data):
    data = raw_data['content'].apply(lambda x: x.replace(u'\u200b', ''))
    data = data.values.tolist()

    doc_clean = [clean(doc).split() for doc in data]

    # model training
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=5, id2word=dictionary, passes=50)

    df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel, corpus=doc_term_matrix, texts=data)
    raw_data['topic'] = df_topic_sents_keywords['Dominant_Topic'].values
    raw_data['topic_keyword'] = df_topic_sents_keywords['Topic_Keywords'].values

    return raw_data

def get_file_names(input_directory):
    result_list = []
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):
            result_list.append(file_name)
    return result_list


def get_data(input_directory, file_list):
    for file_name in file_list:
        path = input_directory + "/" + file_name
        data = pd.read_csv(path)
        data = get_topic(data)
        data.to_csv('data/topic_result/'+file_name, index=False)


def main(inputs):
    dir_name = inputs
    get_data(dir_name, get_file_names(dir_name))
    file_name = 'data/topic_result'
    all_data = get_data(file_name, get_file_names(file_name))
    all_data.to_csv("data/topic_result/topic_all.csv", index=False)


if __name__ == '__main__':
    inputs = sys.argv[1]
    main(inputs)


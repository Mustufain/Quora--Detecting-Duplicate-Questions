import numpy as np
import glob
import multiprocessing  #Multi threading
import os
import re
import nltk
from sklearn.decomposition import PCA
import sklearn.manifold
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gensim
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from fuzzywuzzy import fuzz
import plotly.plotly as py
import string
import numpy as np
from sklearn.preprocessing import scale

def Data_Cleaning(train_file):

    corpus_raw = pd.read_csv(train_file)
    corpus_raw = corpus_raw.dropna()
    corpus_raw['question1'] = corpus_raw['question1'].mask(corpus_raw['question1'] == 0).dropna()
    corpus_raw['question2'] = corpus_raw['question2'].mask(corpus_raw['question2'] == 0).dropna()

    corpus_raw['question1'] = corpus_raw['question1'].apply(lambda row: row.lower())
    corpus_raw['question2'] = corpus_raw['question2'].apply(lambda row: row.lower())


    # Remove Punctuations
    corpus_raw['question1'] = corpus_raw['question1'].apply(lambda row: row.translate(None, string.punctuation))
    corpus_raw['question2'] = corpus_raw['question2'].apply(lambda row: row.translate(None, string.punctuation))

    return corpus_raw

def Feature_set1(corpus_raw):


    #tokenize (not removing stop words)

    corpus_raw['question1_tokens'] = corpus_raw['question1'].apply(lambda row:  row.split())
    corpus_raw['question2_tokens'] = corpus_raw['question2'].apply(lambda row:  row.split())



    # Basic feature Engineering

    corpus_raw['common'] = corpus_raw.apply(
        lambda row: len(list(set(row['question1_tokens']).intersection(row['question2_tokens']))), axis=1)
    corpus_raw['totalwords'] = corpus_raw.apply(
        lambda row: max(len(row['question1_tokens']), len(row['question2_tokens'])), axis=1)

    corpus_raw['proportion_common_words'] = corpus_raw.apply(lambda row: (float(row["common"]) / float(row['totalwords'])) * 100,
                                                             axis=1)
    corpus_raw['text_length_q1'] = corpus_raw.apply(lambda row: len(row['question1_tokens']), axis=1)
    corpus_raw['text_length_q2'] = corpus_raw.apply(lambda row: len(row['question2_tokens']), axis=1)
    corpus_raw['charcount_q1'] = corpus_raw.apply(lambda row: sum([len(char) for char in row['question1_tokens']]),axis=1)
    corpus_raw['charcount_q2'] = corpus_raw.apply(lambda row: sum([len(char) for char in row['question2_tokens']]),axis=1)
    corpus_raw['diff_two_lengths'] = corpus_raw.apply(lambda row: abs(row['text_length_q1'] - row['text_length_q2']),axis=1)

    feature_set1 = ['id','question1', 'question2', 'common', 'totalwords', 'proportion_common_words',
                    'text_length_q1', 'text_length_q2', 'charcount_q1', 'charcount_q2', 'diff_two_lengths','is_duplicate']

    corpus_raw.to_csv("Feature_set1.csv",columns=feature_set1)


def Feature_set2(corpus_raw):


    corpus_raw['fuzz_qratio']=corpus_raw.apply(lambda row : fuzz.QRatio(str(row['question1']),str(row['question2'])),axis=1)
    corpus_raw['fuzz_partialratio'] = corpus_raw.apply(lambda row: fuzz.partial_ratio(str(row['question1']), str(row['question2'])), axis=1)
    corpus_raw['fuzz_partial_token_setratio']=corpus_raw.apply(lambda row : fuzz.partial_token_set_ratio(str(row['question1']),str(row['question2'])),axis=1)
    corpus_raw['fuzz_partial_token_sortratio'] = corpus_raw.apply(lambda row: fuzz.partial_token_sort_ratio(str(row['question1']), str(row['question2'])),axis=1)
    corpus_raw['fuzz_token_setratio'] = corpus_raw.apply(lambda row: fuzz.token_set_ratio(str(row['question1']), str(row['question2'])),axis=1)
    corpus_raw['fuzz_token_sortratio'] = corpus_raw.apply(lambda row: fuzz.token_sort_ratio(str(row['question1']), str(row['question2'])),axis=1)
    corpus_raw['fuzz_wratio'] = corpus_raw.apply(lambda row: fuzz.WRatio(str(row['question1']), str(row['question2'])), axis=1)

    Feature_set2=['id','question1','question2','fuzz_qratio','fuzz_partialratio','fuzz_partial_token_setratio','fuzz_partial_token_sortratio','fuzz_token_sortratio','fuzz_token_setratio','fuzz_wratio','is_duplicate']
    corpus_raw.to_csv('Feature_set2.csv',columns=Feature_set2)

def Exploratory_Data_Analysis():
    # Visualization

    # x-axis----> commonwords(%)
    # y-axis-----> questions(questiondIDs)

    corpus_raw=pd.read_csv("Feature_set1.csv")
    proportion_common_words_is_duplicate = corpus_raw[['proportion_common_words', 'is_duplicate']].groupby(['proportion_common_words'], as_index=False).mean().sort_values(by='is_duplicate', ascending=False)

    #print proportion_common_words_is_duplicate

    x_axis = corpus_raw.loc[corpus_raw['is_duplicate']==0,'proportion_common_words']
    y_axis = corpus_raw.loc[corpus_raw['is_duplicate']==0,'id']


    x_axis1 = corpus_raw.loc[corpus_raw['is_duplicate']==1,'proportion_common_words']
    y_axis1 =  corpus_raw.loc[corpus_raw['is_duplicate']==1,'id']

    labels=['Questions are duplicate','Questions arenot duplicate']

    plt.scatter(x_axis1, y_axis1, label=labels[0], color='blue', marker='s')
    plt.scatter(x_axis,y_axis,label=labels[1],color='red',marker='^')

    plt.legend(loc='lower left', ncol=3, fontsize=12)
    plt.xlabel('Proportion of Common Words')
    plt.ylabel('Question Pairs')
    plt.title("Proportion of Common Words / Question Pairs")


    plt.show()


    #See relatonship between ratios and question pairs


    corpus_raw2=pd.read_csv("Feature_set2.csv")

    #QRatio_is_duplicate = corpus_raw[['fuzz_qratio', 'is_duplicate']].groupby(['fuzz_qratio'], as_index=False).mean().sort_values( by='is_duplicate',ascending=False)

    #x_axis = corpus_raw1.loc[corpus_raw1['is_duplicate'] == 0, 'proportion_common_words']
    #y_axis = corpus_raw.loc[corpus_raw['is_duplicate'] == 0, 'fuzz_partialratio']

    y_axis1 = corpus_raw.loc[corpus_raw['is_duplicate'] == 1, 'proportion_common_words']
    x_axis1 = corpus_raw2.loc[corpus_raw2['is_duplicate'] == 1, 'fuzz_partialratio']
    labels = ['Questions are duplicate', 'Questions arenot duplicate']

    plt.scatter(x_axis1, y_axis1, label=labels[0], color='blue')
    #plt.scatter(x_axis, y_axis, label=labels[1], color='red', marker='^')

    plt.legend(loc='lower left', ncol=3, fontsize=12)
    plt.xlabel('Proportion of Common Words')
    plt.ylabel('Partial Ratio')
    plt.title("Proportion of Common Words / Partial Ratio")

    plt.show()

    plt.hist(x_axis1)
    plt.title("Partial Ratio of duplicate questions")
    plt.xlabel("Partial Ratio")
    plt.ylabel("Frequency")
    plt.show()

    #Feature set 1 is mapped to 2 dimension for visualization using Principal Component Analysis

    clf = PCA(n_components=2)
    data = np.array(corpus_raw.iloc[:, 4:11])
    data = scale(data)
    target = np.array(corpus_raw.iloc[:, 12])
    Transformed_2D = clf.fit(data, target).transform(data)
    corpus_raw['colors'] = corpus_raw.apply(lambda row: 'red' if row['is_duplicate'] == 0 else 'blue', axis=1)

    plt.scatter(Transformed_2D[:, 0], Transformed_2D[:, 1], color=corpus_raw['colors'])

    plt.legend(loc='lower left', ncol=3, fontsize=7)
    plt.title("Feature set 1 project on 2 dimension")
    plt.show()

    # Feature set 2 is mapped to 2 dimension for visualization using Principal Component Analysis

    clf = PCA(n_components=2)
    data = np.array(corpus_raw2.iloc[:, 4:10])
    data = scale(data)
    target = np.array(corpus_raw2.iloc[:, 11])
    Transformed_2D = clf.fit(data, target).transform(data)

    corpus_raw2['colors'] = corpus_raw2.apply(lambda row: 'red' if row['is_duplicate'] == 0 else 'blue', axis=1)

    plt.scatter(Transformed_2D[:, 0], Transformed_2D[:, 1], color=corpus_raw2['colors'])

    plt.legend(loc='lower left', ncol=3, fontsize=7)
    plt.title("Feature set 2 project on 2 dimension")
    plt.show()






def Main():

    file='train.csv'
    corpus_raw=Data_Cleaning(file)
    Feature_set1(corpus_raw)
    Feature_set2(corpus_raw)
    Exploratory_Data_Analysis()

    
Main()
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

def Feature_set1():

    corpus_raw = pd.read_csv("train.csv")
    corpus_raw = corpus_raw.dropna()
    stop = stopwords.words('english')
    reg = "[^a-zA-Z]"

    corpus_raw['question1'] = corpus_raw['question1'].apply(lambda x: re.sub(reg, " ", x))
    corpus_raw['question2'] = corpus_raw['question2'].apply(lambda x: re.sub(reg, " ", x))

    # Basic feature Engineering

    corpus_raw['question1_tokens'] = corpus_raw['question1'].apply(lambda x: [x for x in x.split() if x not in stop])
    corpus_raw['question2_tokens'] = corpus_raw['question2'].apply(lambda x: [x for x in x.split() if x not in stop])
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

    feature_set1 = ['question1_tokens', 'question2_tokens', 'common', 'totalwords', 'proportion_common_words',
                    'text_length_q1', 'text_length_q2', 'charcount_q1', 'charcount_q2', 'diff_two_lengths','is_duplicate']

    corpus_raw.to_csv("Feature_set1.csv",columns=feature_set1)


def Feature_set2():

    corpus_raw = pd.read_csv('train.csv')

    corpus_raw['fuzz_qratio']=corpus_raw.apply(lambda row : fuzz.QRatio(str(row['question1']),str(row['question2'])),axis=1)
    corpus_raw['fuzz_partialratio'] = corpus_raw.apply(lambda row: fuzz.partial_ratio(str(row['question1']), str(row['question2'])), axis=1)
    corpus_raw['fuzz_partial_token_setratio']=corpus_raw.apply(lambda row : fuzz.partial_token_set_ratio(str(row['question1']),str(row['question2'])),axis=1)
    corpus_raw['fuzz_partial_token_sortratio'] = corpus_raw.apply(lambda row: fuzz.partial_token_sort_ratio(str(row['question1']), str(row['question2'])),axis=1)
    corpus_raw['fuzz_token_setratio'] = corpus_raw.apply(lambda row: fuzz.token_set_ratio(str(row['question1']), str(row['question2'])),axis=1)
    corpus_raw['fuzz_token_sortratio'] = corpus_raw.apply(lambda row: fuzz.token_sort_ratio(str(row['question1']), str(row['question2'])),axis=1)
    corpus_raw['fuzz_wratio'] = corpus_raw.apply(lambda row: fuzz.WRatio(str(row['question1']), str(row['question2'])), axis=1)

    Feature_set2=['id','fuzz_qratio','fuzz_partialratio','fuzz_partial_token_setratio','fuzz_partial_token_sortratio','fuzz_token_sortratio','fuzz_token_setratio','fuzz_wratio','is_duplicate']

    corpus_raw.to_csv('Feature_set2.csv',columns=Feature_set2)

def Exploratory_Data_Analysis():
    # Visualization

    # x-axis----> commonwords(%)
    # y-axis-----> questions(questiondIDs)
    corpus_raw=pd.read_csv("Feature_set1.csv")
    proportion_common_words_is_duplicate = corpus_raw[['proportion_common_words', 'is_duplicate']].groupby(['proportion_common_words'], as_index=False).mean().sort_values(
        by='is_duplicate',
        ascending=False)

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

    corpus_raw1=pd.read_csv("Feature_set1.csv")
    corpus_raw=pd.read_csv("Feature_set2.csv")

    QRatio_is_duplicate = corpus_raw[['fuzz_qratio', 'is_duplicate']].groupby(
        ['fuzz_qratio'], as_index=False).mean().sort_values(
        by='is_duplicate',
        ascending=False)

    #x_axis = corpus_raw1.loc[corpus_raw1['is_duplicate'] == 0, 'proportion_common_words']
    #y_axis = corpus_raw.loc[corpus_raw['is_duplicate'] == 0, 'fuzz_partialratio']

    y_axis1 = corpus_raw1.loc[corpus_raw1['is_duplicate'] == 1, 'proportion_common_words']
    x_axis1 = corpus_raw.loc[corpus_raw['is_duplicate'] == 1, 'fuzz_partialratio']
    labels = ['Questions are duplicate', 'Questions arenot duplicate']

    plt.scatter(x_axis1, y_axis1, label=labels[0], color='blue')
    #plt.scatter(x_axis, y_axis, label=labels[1], color='red', marker='^')

    plt.legend(loc='lower left', ncol=3, fontsize=12)
    plt.xlabel('Proportion of Common Words')
    plt.ylabel('Partial Ratio')
    plt.title("Proportion of Common Words / Partial Ratio")

    plt.show()


# def words_to_vectors():[
#     #Once we have vectors
#     #distance, simalrity, Ranking
#
#
#     num_features=300  #Dimensionality of vectors
#     num_workers = multiprocessing.cpu_count()  #num of threads running in parallel so that it would run faster
#     context_size=7           #window size to look number of words in context of the given word
#     downsampling=1e-3 #Any number between 0 and 1e-5 is good for this (doesnot look repeatedly at the same words)
#     seed=1      #random number generator what part of the text we want to look and vectorize it
#     min_word_count=3    #smallest set of words we want to recognize when converting into vector
#
#     thrones2vec = gensim.models.Word2Vec(
#     sg=1,
#     seed=1,
#     workers=num_workers,
#     size=num_features,
#     window=context_size,
#     min_count=min_word_count,
#     sample=downsampling
#  )
#
#     #thrones2vec.build_vocab(wordlist)
#     #thrones2vec.train(wordlist)
#     #if not os.path.exists("trained"):
#         #os.makedirs("trained")
#     #thrones2vec.save(os.path.join("trained","thrones2vec.w2v"))
#
# def Visualization():
#
#     thrones2vec=gensim.models.KeyedVectors.load_word2vec_format(os.path.join("trained",'GoogleNews-vectors-negative300.bin'), binary=True)
#     all_word_vectors_matrix = thrones2vec.wv.syn0   #300 features
#     print (all_word_vectors_matrix.shape)
#
#     #clf=PCA(n_components=2)                                                #PCA is performed to visualize only
#     #all_word_vectors_matrix_2D=clf.fit_transform(all_word_vectors_matrix)
#
#
#
# 
#     #clf = sklearn.manifold.TSNE(n_components=2,random_state=0)
#     #all_word_vectors_matrix_2D=clf.fit_transform(all_word_vectors_matrix)
#
#     #Average all the vectors in a sentence
#



def Main():

    Feature_set1()
    Feature_set2()
    Exploratory_Data_Analysis()
    #words_to_vectors()

    
Main()
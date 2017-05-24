import numpy as np
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import string
import numpy as np
import time
from math import *
from scipy import spatial
from scipy import stats
import gensim
import multiprocessing
import string
import logging
import sys

def Data_Cleaning(data):

    corpus_raw = pd.read_csv(data)

    corpus_raw['question1'] = corpus_raw['question1'].apply(lambda row: str(row).lower())
    corpus_raw['question2'] = corpus_raw['question2'].apply(lambda row: str(row).lower())





    # preserve contractions and ingore punctuations occuring in sentences--- remove punctuation between words except "'"
    corpus_raw['question1'] = corpus_raw['question1'].apply(lambda row: "".join([s if s.isalpha()  or s.isdigit() or s=="'" or s==" " else ' ' for s in row ]))
    corpus_raw['question2'] = corpus_raw['question2'].apply(lambda row: "".join([s  if s.isalpha() or s.isdigit() or s == "'" or s == " " else ' ' for s in row]))

    stop = stopwords.words('english')

    # Remove stopwords , numeric and alphanumeric characters

    corpus_raw['question1_tokens'] = corpus_raw['question1'].apply(
        lambda row: [row for row in row.split() if row not in stop and row.isalpha()])
    corpus_raw['question2_tokens'] = corpus_raw['question2'].apply(

        lambda row: [row for row in row.split() if row not in stop and row.isalpha()])

    corpus_raw['question1_tokens'] = corpus_raw['question1_tokens'].apply(lambda row: ",".join(row))
    corpus_raw['question2_tokens'] = corpus_raw['question2_tokens'].apply(lambda row: ",".join(row))




    return corpus_raw

#----------------------------------------Basic Features-----------------------------------------------------------------------------#

def Feature_set1(corpus_raw):

    corpus_raw['common'] = corpus_raw.apply(
        lambda row: len(list(set(row['question1_tokens']).intersection(row['question2_tokens']))), axis=1)
    corpus_raw['totalwords'] = corpus_raw.apply(
        lambda row: max(len(row['question1_tokens']), len(row['question2_tokens'])), axis=1)

    corpus_raw['text_length_q1'] = corpus_raw.apply(lambda row: len(row['question1_tokens']), axis=1)
    corpus_raw['text_length_q2'] = corpus_raw.apply(lambda row: len(row['question2_tokens']), axis=1)
    corpus_raw['charcount_q1'] = corpus_raw.apply(lambda row: sum([len(char) for char in row['question1_tokens']]),
                                                  axis=1)
    corpus_raw['charcount_q2'] = corpus_raw.apply(lambda row: sum([len(char) for char in row['question2_tokens']]),
                                                  axis=1)
    corpus_raw['diff_two_lengths'] = corpus_raw.apply(lambda row: abs(row['text_length_q1'] - row['text_length_q2']),
                                                      axis=1)

#-----------------------------------------------------------------Laveinshtein distances----------------------------------------------#

def Feature_set2(corpus_raw):

    corpus_raw['fuzz_qratio'] = corpus_raw.apply(lambda row: fuzz.QRatio(str(row['question1']), str(row['question2'])),
                                                 axis=1)
    corpus_raw['fuzz_partialratio'] = corpus_raw.apply(
        lambda row: fuzz.partial_ratio(str(row['question1']), str(row['question2'])), axis=1)
    corpus_raw['fuzz_partial_token_setratio'] = corpus_raw.apply(
        lambda row: fuzz.partial_token_set_ratio(str(row['question1']), str(row['question2'])), axis=1)
    corpus_raw['fuzz_partial_token_sortratio'] = corpus_raw.apply(
        lambda row: fuzz.partial_token_sort_ratio(str(row['question1']), str(row['question2'])), axis=1)
    corpus_raw['fuzz_token_setratio'] = corpus_raw.apply(
        lambda row: fuzz.token_set_ratio(str(row['question1']), str(row['question2'])), axis=1)
    corpus_raw['fuzz_token_sortratio'] = corpus_raw.apply(
        lambda row: fuzz.token_sort_ratio(str(row['question1']), str(row['question2'])), axis=1)
    corpus_raw['fuzz_wratio'] = corpus_raw.apply(lambda row: fuzz.WRatio(str(row['question1']), str(row['question2'])),
                                                 axis=1)

#------------------------------------------------------------------------------------------------------------------------------------#
def words_to_vectors(fs1,filename):
    # Once we have vectors

    # distance, simalrity, Ranking



    num_features = 300  # Dimensionality of vectors
    num_workers = multiprocessing.cpu_count()  # num of threads running in parallel so that it would run faster
    context_size = 5  # window size to look number of words in context of the given word
    downsampling = 7.5e-06  # Any number between 0 and 1e-5 is good for this (doesnot look repeatedly at the same words)
    seed = 1  # random number generator what part of the text we want to look and vectorize it
    min_word_count = 5  #  ignore all words with total frequency lower than this.
    hs = 1  #if 1, hierarchical softmax will be used for model training.
            # If set to 0 (default), and negative is non-zero, negative sampling will be used.
    negative = 5 #if > 0, negative sampling will be used, the int for negative specifies how many "noise words"
                 # should be drawn (usually between 5-20). Default is 5. If set to 0, no negative samping is used


    Quora_word2vec = gensim.models.Word2Vec(

        sg=0,
        seed=1,
        workers=num_workers,
        min_count=min_word_count,
        size=num_features,
        window=context_size, #(5)
        hs=hs,#(1)
        negative = negative,  #(5)
        sample=downsampling #(7.5e-06 )

         )

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    Quora_word2vec = gensim.models.KeyedVectors.load_word2vec_format('wiki.en.vec', binary=False)

    # Distance column

    fs1['euclidean'] = ''
    fs1['canberra'] = ''
    fs1['wordmover'] = ''
    fs1['normwordmover'] = ''
    fs1['kurtosisq1'] = ''
    fs1['kurtosisq2'] = ''
    fs1['skewq1'] = ''
    fs1['skewq2'] = ''
    fs1['manhattan'] = ''
    fs1['minkowksi'] = ''
    fs1['jaccard'] = ''
    fs1['cossimilarity'] = ''
    fs1['braycurtis'] = ''

    # Feature list
    Features = ['id', 'common', 'totalwords','text_length_q1', 'text_length_q2',
                'charcount_q1', 'charcount_q2', 'diff_two_lengths', 'fuzz_qratio', 'fuzz_partialratio',
                'fuzz_partial_token_setratio', 'fuzz_partial_token_sortratio', 'fuzz_token_sortratio',
                'fuzz_token_setratio',
                'fuzz_wratio', 'euclidean', 'manhattan', 'kurtosisq1', 'kurtosisq2',
                'skewq1', 'skewq2', 'canberra', 'braycurtis', 'minkowski',
                'wordmover', 'normwordmover', 'jaccard', 'cossimilarity']

    q1vector = np.zeros(shape=(fs1.shape[0], num_features))
    q2vector = np.zeros(shape=(fs1.shape[0], num_features))

    for index, row in fs1.iterrows():

        # Words found in vocabulary

        q1_word = filter(lambda word: word in Quora_word2vec.vocab, row['question1_tokens'].split(','))
        q2_word = filter(lambda word: word in Quora_word2vec.vocab, row['question2_tokens'].split(','))



        q1_vector, q2_vector = get_vector(q1_word, q2_word, Quora_word2vec,num_features)
        q1vector[index]=q1_vector
        q2vector[index]=q2_vector
        Distance_Functions(Quora_word2vec, fs1, index, q1_vector, q2_vector, row['question1_tokens'],row['question2_tokens'],num_features)

    for i in range(0, num_features):
        fs1['q1_vector_' + str(i)] = q1vector[:, i]

    for i in range(0,num_features):
        fs1['q2_vector_' + str(i)] = q2vector[:,i]


    Normalized_Word_Mover_Distance(fs1, Quora_word2vec)





    # Write into file here
    Write_to_csv(fs1, Features,num_features,filename)

    
def get_vector(q1_word, q2_word, model,num_features):
    q1_vector_word = []
    q2_vector_word = []

    if len(q1_word) == 0 and len(q2_word) > 0:  # if q1 not found in vocabulary

        for word in q2_word:
            q2_vector_word.append(model[word])  # Vector representation of each word in q1

        q2_vector = np.array(np.sum(q2_vector_word, axis=0) / np.sqrt((np.sum(q2_vector_word, axis=0) ** 2).sum()))
        # q1_vector = np.zeros(shape=q2_vector.shape)
        q1_vector = np.empty(shape=(q2_vector.shape))
        q1_vector[:] = np.nan

    if len(q2_word) == 0 and len(q1_word) > 0:  # if q2 not found in vocabulary

        for word in q1_word:
            q1_vector_word.append(model[word])  # Vector representation of each word in q2

        q1_vector = np.array(np.sum(q1_vector_word, axis=0) / np.sqrt((np.sum(q1_vector_word, axis=0) ** 2).sum()))
        # q2_vector = np.zeros(shape=q1_vector.shape)
        q2_vector = np.empty(shape=(q1_vector.shape))
        q2_vector[:] = np.nan

    if len(q1_word) == 0 and len(q2_word) == 0:

        q1_vector = np.empty(shape=(num_features, 1))
        q2_vector = np.empty(shape=q1_vector.shape)
        q1_vector[:] = np.nan
        q2_vector[:] = np.nan

    if len(q1_word) > 0 and len(q2_word) > 0:  # If both are found in vocabulary

        for word in q1_word:
            q1_vector_word.append(model[word])  # Vector representation of each word in q1

        for word in q2_word:
            q2_vector_word.append(model[word])  # vector representation of each word in q2

        # vector representation of q1 && q2
        q1_vector = np.array(np.sum(q1_vector_word, axis=0) / np.sqrt((np.sum(q1_vector_word, axis=0) ** 2).sum()))
        q2_vector = np.array(np.sum(q2_vector_word, axis=0) / np.sqrt((np.sum(q2_vector_word, axis=0) ** 2).sum()))

    # row vectors (1,num_features)

    q1_vector = q1_vector.reshape(1, q1_vector.shape[0])

    q2_vector = q2_vector.reshape(1, q2_vector.shape[0])

    print q1_vector.shape
    print q2_vector.shape

    return q1_vector, q2_vector


def Distance_Functions(model, fs1, index, q1_vector, q2_vector, q1, q2,num_features):
    try:
        fs1.set_value(index, 'euclidean', Euclidean_Distance(q1_vector, q2_vector))

    except Exception as e:


        fs1.set_value(index, 'euclidean', np.nan)

    try:
        fs1.set_value(index, 'manhattan', Manhattan(q1_vector, q2_vector))
    except Exception as e:

        fs1.set_value(index, 'manhattan', np.nan)
    try:
        fs1.set_value(index, 'jaccard', Jaccard(q1_vector, q2_vector))
    except Exception as e:

        fs1.set_value(index, 'jaccard', np.nan)
    try:
        fs1.set_value(index, 'cossimilarity', Cosine_Similarity(q2_vector, q2_vector))
    except Exception as e:

        fs1.set_value(index, 'cossimilarity', np.nan)
    try:
        fs1.set_value(index, 'canberra', Canberra(q1_vector, q2_vector))
    except Exception as e:


        fs1.set_value(index, 'canberra', np.nan)
    try:
        fs1.set_value(index, 'braycurtis', Braycurtis(q1_vector, q2_vector))
    except Exception as e:

        fs1.set_value(index, 'braycurtis', np.nan)
    try:
        fs1.set_value(index, 'minkowski', Minkowksi(q1_vector, q2_vector, 3))
    except Exception as e:

        fs1.set_value(index, 'minkowski', np.nan)
    try:
        fs1.set_value(index, 'wordmover', Word_Mover_Distance(q1, q2, model))
    except Exception as e:

        fs1.set_value(index, 'wordmover', np.nan)
    try:
        fs1.set_value(index, 'kurtosisq1', Kurtosis_q1_vector(q1_vector,num_features))
    except Exception as e:

        fs1.set_value(index, 'kurtosisq1', np.nan)
    try:
        fs1.set_value(index, 'kurtosisq2', Kurtosis_q2_vector(q2_vector,num_features))
    except Exception as e:

        fs1.set_value(index, 'kurtosisq2', np.nan)
    try:
        fs1.set_value(index, 'skewq1', Skew_q1_vector(q1_vector,num_features))
    except Exception as e:

        fs1.set_value(index, 'skewq1', np.nan)
    try:
        fs1.set_value(index, 'skewq2', Skew_q2_vector(q2_vector,num_features))
    except Exception as e:

        fs1.set_value(index, 'skewq2', np.nan)


    # ---------------------------------------------------------Write all features into csv file------------------------------------------#


def Write_to_csv(fs1, Features,num_feaures,filename):

    for i in range(0, num_feaures):
        Features.append('q1_vector_' + str(i))

    for i in range(0, num_feaures):
        Features.append('q2_vector_' + str(i))

    if filename == 'train.csv':

        Features.append('is_duplicate')
        fs1.to_csv("Quora_Features_" + str(filename) + ".csv", columns=Features, index=False)

    else:

        fs1.to_csv("Quora_Features_"+str(filename)+".csv", columns=Features,index=False)

    #log parameters matching the file variant



# -----------------------------------------Features of vectors------------------------------------------------------------------#-


def Manhattan(q1_vector, q2_vector):
    return spatial.distance.cityblock(q1_vector, q2_vector)


def Cosine_Similarity(q1_vector, q2_vector):
    return spatial.distance.cosine(q1_vector, q2_vector)


def Euclidean_Distance(q1_vector, q2_vector):
    return spatial.distance.euclidean(q1_vector, q2_vector)


def Minkowksi(q1_vector, q2_vector, p):
    return spatial.distance.minkowski(q1_vector, q2_vector, p)


def Canberra(q1_vector, q2_vector):
    return spatial.distance.canberra(q1_vector, q2_vector)


def Braycurtis(q1_vector, q2_vector):
    return spatial.distance.braycurtis(q1_vector, q2_vector)


def Jaccard(q1_vector, q2_vector):
    return spatial.distance.jaccard(q1_vector, q2_vector)


def Kurtosis_q1_vector(q1_vector,num_features):
    kurt_q1 = stats.kurtosis(q1_vector.reshape(num_features, 1))[0]
    return kurt_q1


def Kurtosis_q2_vector(q2_vector,num_features):
    kurt_q2 = stats.kurtosis(q2_vector.reshape(num_features, 1))[0]
    return kurt_q2


def Skew_q1_vector(q1_vector,num_features):
    skew_q1 = stats.skew(q1_vector.reshape(num_features, 1))[0]
    return skew_q1


def Skew_q2_vector(q2_vector,num_features):
    skew_q2 = stats.skew(q2_vector.reshape(num_features, 1))[0]
    return skew_q2


def Word_Mover_Distance(q1, q2, model):
    return model.wmdistance(q1, q2)


def Normalized_Word_Mover_Distance(fs1, model):
    model.init_sims(replace=True)
    temp = np.zeros(shape=fs1['question1_tokens'].shape[0])

    for index, row in fs1.iterrows():

        try:

            temp = model.wmdistance(fs1.loc[index, 'question1_tokens'], fs1.loc[index, 'question2_tokens'])
            fs1.set_value(index, 'normwordmover', temp)

        except Exception as e:

            fs1.set_value(index,'normwordmover',np.nan)



def Main():

    start_time = time.time()
    data = sys.argv[1]                #file
    filtered_data = Data_Cleaning(data)
    Feature_set1(filtered_data)
    Feature_set2(filtered_data)
                   
    words_to_vectors(filtered_data,data)
                 
    print("--- %s seconds ---" % (time.time() - start_time))


Main()

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
def words_to_vectors(fs1):
    # Once we have vectors

    # distance, simalrity, Ranking


    num_features = 300  # Dimensionality of vectors
    num_workers = multiprocessing.cpu_count()  # num of threads running in parallel so that it would run faster
    context_size = 5  # window size to look number of words in context of the given word
    downsampling = 1e-3  # Any number between 0 and 1e-5 is good for this (doesnot look repeatedly at the same words)
    seed = 1  # random number generator what part of the text we want to look and vectorize it
    min_word_count = 5  # smallest set of words we want to recognize when converting into vector

    Quora_word2vec = gensim.models.Word2Vec(
        sg=1,
        seed=1,
        workers=num_workers,
        size=num_features,
        window=context_size,
        min_count=min_word_count,
        sample=downsampling

    )

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    Quora_word2vec = gensim.models.KeyedVectors.load_word2vec_format('wiki.en.vec', binary=False)

    # There are for keeping track which words are missed

    fs1['q1_words_not_in_vocab'] = ''
    fs1['q2_words_not_in_vocab'] = ''
    fs1['q1_words_in_vocab'] = ''
    fs1['q2_words_in_vocab'] = ''

    # There are just for analysis

    fs1['q1textlength'] = ''
    fs1['q2textlength'] = ''
    fs1['q1word_in_vocab_length'] = ''
    fs1['q2word_in_vocab_length'] = ''
    fs1['q1word_not_in_vocab_length'] = ''
    fs1['q2word_not_in_vocab_length'] = ''

    # Distance columns

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

    Missing_Features = ['id', 'question1', 'question2', 'question1_tokens', 'question2_tokens', 'q1textlength',
                        'q2textlength',
                        'q1word_in_vocab_length', 'q2word_in_vocab_length', 'q1word_not_in_vocab_length',
                        'q2word_not_in_vocab_length',
                        'q1_words_in_vocab', 'q2_words_in_vocab', 'q1_words_not_in_vocab', 'q2_words_not_in_vocab',
                        'is_duplicate']

    for index, row in fs1.iterrows():
        # filter(function, iterable) is equivalent to[item for item in iterable if function(item)]





        # Words found in vocabulary

        q1_word = filter(lambda word: word in Quora_word2vec.vocab, row['question1_tokens'].split(','))
        q2_word = filter(lambda word: word in Quora_word2vec.vocab, row['question2_tokens'].split(','))

        # words not found in vocabulary

        missing_q1_word = filter(lambda word: word not in q1_word, row['question1_tokens'].split(','))
        missing_q2_word = filter(lambda word: word not in q2_word, row['question2_tokens'].split(','))

        # There features are just for analysis to see proportion of words missed in each pair in wikipedia corpus
        fs1.set_value(index, 'q1textlength', len(row['question1_tokens'].split(',')))
        fs1.set_value(index, 'q2textlength', len(row['question2_tokens'].split(',')))
        fs1.set_value(index, 'q1word_in_vocab_length', len(q1_word))
        fs1.set_value(index, 'q2word_in_vocab_length', len(q2_word))
        fs1.set_value(index, 'q1word_not_in_vocab_length', len(missing_q1_word))
        fs1.set_value(index, 'q2word_not_in_vocab_length', len(missing_q2_word))

        # Main work starts from here
        q1_vector, q2_vector = get_vector(q1_word, q2_word, Quora_word2vec)
        Questions_In_Vocab(fs1, row, index, q1_vector, q2_vector, q1_word, q2_word)
        Missing_Words_In_Question_In_Vocab(fs1, missing_q1_word, missing_q2_word, index)
        Distance_Functions(Quora_word2vec, fs1, index, q1_vector, q2_vector, row['question1_tokens'],
                           row['question2_tokens'])


    Normalized_Word_Mover_Distance(fs1, Quora_word2vec)

    # Write into file here
    Write_to_csv(fs1, Features, Missing_Features)


def get_vector(q1_word, q2_word, model):
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
        # q1_vector=np.zeros(shape=(300,1))
        # q2_vector=np.zeros(shape=q1_vector.shape)
        q1_vector = np.empty(shape=(300, 1))
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

    # row vectors (1,300)

    q1_vector = q1_vector.reshape(1, q1_vector.shape[0])

    q2_vector = q2_vector.reshape(1, q2_vector.shape[0])

    return q1_vector, q2_vector


def Distance_Functions(model, fs1, index, q1_vector, q2_vector, q1, q2):
    try:
        fs1.set_value(index, 'euclidean', Euclidean_Distance(q1_vector, q2_vector))

    except Exception as e:

        print e
        fs1.set_value(index, 'euclidean', np.nan)

    try:
        fs1.set_value(index, 'manhattan', Manhattan(q1_vector, q2_vector))
    except Exception as e:
        print e
        fs1.set_value(index, 'manhattan', np.nan)
    try:
        fs1.set_value(index, 'jaccard', Jaccard(q1_vector, q2_vector))
    except Exception as e:
        print e
        fs1.set_value(index, 'jaccard', np.nan)
    try:
        fs1.set_value(index, 'cossimilarity', Cosine_Similarity(q2_vector, q2_vector))
    except Exception as e:
        print e
        fs1.set_value(index, 'cossimilarity', np.nan)
    try:
        fs1.set_value(index, 'canberra', Canberra(q1_vector, q2_vector))
    except Exception as e:

        print e
        fs1.set_value(index, 'canberra', np.nan)
    try:
        fs1.set_value(index, 'braycurtis', Braycurtis(q1_vector, q2_vector))
    except Exception as e:
        print e
        fs1.set_value(index, 'braycurtis', np.nan)
    try:
        fs1.set_value(index, 'minkowski', Minkowksi(q1_vector, q2_vector, 3))
    except Exception as e:
        print e
        fs1.set_value(index, 'minkowski', np.nan)
    try:
        fs1.set_value(index, 'wordmover', Word_Mover_Distance(q1, q2, model))
    except Exception as e:
        print e
        fs1.set_value(index, 'wordmover', np.nan)
    try:
        fs1.set_value(index, 'kurtosisq1', Kurtosis_q1_vector(q1_vector))
    except Exception as e:
        print e
        fs1.set_value(index, 'kurtosisq1', np.nan)
    try:
        fs1.set_value(index, 'kurtosisq2', Kurtosis_q2_vector(q2_vector))
    except Exception as e:
        print e
        fs1.set_value(index, 'kurtosisq2', np.nan)
    try:
        fs1.set_value(index, 'skewq1', Skew_q1_vector(q1_vector))
    except Exception as e:
        print e
        fs1.set_value(index, 'skewq1', np.nan)
    try:
        fs1.set_value(index, 'skewq2', Skew_q2_vector(q2_vector))
    except Exception as e:
        print e
        fs1.set_value(index, 'skewq2', np.nan)


    # ---------------------------------------------------------Write all features into csv file------------------------------------------#


def Write_to_csv(fs1, Features, Missing_Features):
    for i in range(0, 300):
        Features.append('q1_vector_' + str(i))

    for i in range(0, 300):
        Features.append('q2_vector_' + str(i))

    Features.append('is_duplicate')

    # Word 2 vec features

    fs1.to_csv("Quora_Engineered_Features.csv", columns=Features)

    # Missing words in word2vec-----> for analysis

    fs1.to_csv("missing_words_word2vec.csv", columns=Missing_Features)


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


def Kurtosis_q1_vector(q1_vector):
    kurt_q1 = stats.kurtosis(q1_vector.reshape(300, 1))[0]
    return kurt_q1


def Kurtosis_q2_vector(q2_vector):
    kurt_q2 = stats.kurtosis(q2_vector.reshape(300, 1))[0]
    return kurt_q2


def Skew_q1_vector(q1_vector):
    skew_q1 = stats.skew(q1_vector.reshape(300, 1))[0]
    return skew_q1


def Skew_q2_vector(q2_vector):
    skew_q2 = stats.skew(q2_vector.reshape(300, 1))[0]
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

    # -------------------------------------------Only for analysis which are not found in wordvec---------------------------------------#


def Questions_In_Vocab(fs1, row, count_found, q1_vector, q2_vector, q1_word, q2_word):
    for i in range(0, q1_vector.shape[1]):
        fs1.set_value(count_found, 'q1_vector_' + str(i), q1_vector[:, i])

    for i in range(0, q1_vector.shape[1]):
        fs1.set_value(count_found, 'q2_vector_' + str(i), q2_vector[:, i])

    if len(q1_word) > 1:

        fs1.set_value(count_found, 'q1_words_in_vocab', ",".join(q1_word))

    elif len(q1_word) == 1:

        fs1.set_value(count_found, 'q1_words_in_vocab', q1_word[0])

    elif len(q1_word) == 0:

        fs1.set_value(count_found, 'q1_words_in_vocab', 0)

    if len(q2_word) > 1:

        fs1.set_value(count_found, 'q2_words_in_vocab', ",".join(q2_word))

    elif len(q2_word) == 1:

        fs1.set_value(count_found, 'q2_words_in_vocab', q2_word[0])

    elif len(q2_word) == 0:

        fs1.set_value(count_found, 'q2_words_in_vocab', 0)


def Missing_Words_In_Question_In_Vocab(fs1, missing_q1_word, missing_q2_word, count_found):
    if len(missing_q1_word) > 1:

        fs1.set_value(count_found, 'q1_words_not_in_vocab', ",".join(missing_q1_word))

    elif len(missing_q1_word) == 1:

        fs1.set_value(count_found, 'q1_words_not_in_vocab', missing_q1_word[0])

    elif len(missing_q1_word) == 0:

        fs1.set_value(count_found, 'q1_words_not_in_vocab', 0)

    if len(missing_q2_word) > 1:

        fs1.set_value(count_found, 'q2_words_not_in_vocab', ",".join(missing_q2_word))

    elif len(missing_q2_word) == 1:

        fs1.set_value(count_found, 'q2_words_not_in_vocab', missing_q2_word[0])

    elif len(missing_q2_word) == 0:

        fs1.set_value(count_found, 'q2_words_not_in_vocab', 0)


def Main():
    start_time = time.time()
    data = 'train.csv'
    filtered_data = Data_Cleaning(data)
    Feature_set1(filtered_data)
    Feature_set2(filtered_data)
    words_to_vectors(filtered_data)
    print("--- %s seconds ---" % (time.time() - start_time))


Main()

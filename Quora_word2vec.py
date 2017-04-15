import gensim
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing  #Multi threading
import sklearn.manifold
import sys
import logging
import numpy as np
import string
from nltk.corpus import stopwords
from math import *
from scipy import spatial
from scipy import stats

#I am using pre trained word2vec on google news corpus


def Data_Cleaning(train_file):

    corpus_raw = pd.read_csv(train_file,nrows=5)
    corpus_raw = corpus_raw.dropna()
    corpus_raw['question1'] = corpus_raw['question1'].mask(corpus_raw['question1'] == 0).dropna()
    corpus_raw['question2'] = corpus_raw['question2'].mask(corpus_raw['question2'] == 0).dropna()

    corpus_raw['question1'] = corpus_raw['question1'].apply(lambda row: row.lower())
    corpus_raw['question2'] = corpus_raw['question2'].apply(lambda row: row.lower())

    # Remove Punctuations
    corpus_raw['question1'] = corpus_raw['question1'].apply(lambda row: row.translate(None, string.punctuation))
    corpus_raw['question2'] = corpus_raw['question2'].apply(lambda row: row.translate(None, string.punctuation))

    stop=stopwords.words('english')

    #Remove stopwords

    corpus_raw['question1_tokens'] = corpus_raw['question1'].apply(
    lambda row: [row for row in row.split()])
    corpus_raw['question2_tokens'] = corpus_raw['question2'].apply(

    lambda row: [row for row in row.split()])

    corpus_raw['question1_tokens'] = corpus_raw['question1_tokens'].apply(lambda row: ",".join(row))
    corpus_raw['question2_tokens'] = corpus_raw['question2_tokens'].apply(lambda row: ",".join(row))

    print corpus_raw.isnull().sum()
    return corpus_raw

def words_to_vectors(fs1):

    #Once we have vectors

    #distance, simalrity, Ranking


    num_features=300                            #Dimensionality of vectors
    num_workers = multiprocessing.cpu_count()   #num of threads running in parallel so that it would run faster
    context_size=5                              #window size to look number of words in context of the given word
    downsampling=1e-3                            #Any number between 0 and 1e-5 is good for this (doesnot look repeatedly at the same words)
    seed=1                                      #random number generator what part of the text we want to look and vectorize it
    min_word_count=5                             #smallest set of words we want to recognize when converting into vector

    Quora_word2vec= gensim.models.Word2Vec(
    sg=1,
    seed=1,
    workers=num_workers,
    size=num_features,
    window=context_size,
    min_count=min_word_count,
    sample=downsampling

     )

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    Quora_word2vec=gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)

    #There are for keeping track which words are missed

    fs1['q1_words_not_in_vocab']=''
    fs1['q2_words_not_in_vocab']=''
    fs1['q1_words_in_vocab']=''
    fs1['q2_words_in_vocab']=''

    #There are just for analysis

    fs1['q1textlength']=''
    fs1['q2textlength']=''
    fs1['q1word_in_vocab_length']=''
    fs1['q2word_in_vocab_length']=''
    fs1['q1word_not_in_vocab_length']=''
    fs1['q2word_not_in_vocab_length']=''

    #Distance columns

    fs1['euclidean']=''
    fs1['canberra']=''
    fs1['wordmover']=''
    fs1['normwordmover']=''
    fs1['kurtosisq1']=''
    fs1['kurtosisq2']=''
    fs1['skewq1']=''
    fs1['skewq2']=''
    fs1['manhattan']=''
    fs1['minkowksi']=''
    fs1['jaccard']=''
    fs1['cossimilarity']=''
    fs1['braycurtis']=''

    #Feature list
    Features=['id','euclidean','manhattan','kurtosisq1','kurtosisq2','skewq1','skewq2','canberra','braycurtis','minkowski',
              'wordmover','normwordmover','jaccard','cossimilarity']

    Missing_Features=['id','question1','question2','question1_tokens','question2_tokens','q1textlength','q2textlength',
              'q1word_in_vocab_length','q2word_in_vocab_length','q1word_not_in_vocab_length','q2word_not_in_vocab_length',
              'q1_words_in_vocab','q2_words_in_vocab','q1_words_not_in_vocab','q2_words_not_in_vocab','is_duplicate']

    for index,row in fs1.iterrows():
        #filter(function, iterable) is equivalent to[item for item in iterable if function(item)]





        # Words found in vocabulary

        q1_word=filter(lambda word : word  in Quora_word2vec.vocab,row['question1_tokens'].split(','))
        q2_word = filter(lambda word: word in Quora_word2vec.vocab, row['question2_tokens'].split(','))


        #words not found in vocabulary

        missing_q1_word = filter(lambda word: word not in  q1_word,row['question1_tokens'].split(','))
        missing_q2_word = filter(lambda word: word not in q2_word,row['question2_tokens'].split(','))

        #There features are just for analysis to see proportion of words missed in each pair in word2vec
        fs1.set_value(index,'q1textlength',len(row['question1_tokens'].split(',')))
        fs1.set_value(index, 'q2textlength', len(row['question2_tokens'].split(',')))
        fs1.set_value(index, 'q1word_in_vocab_length', len(q1_word))
        fs1.set_value(index, 'q2word_in_vocab_length', len(q2_word))
        fs1.set_value(index, 'q1word_not_in_vocab_length', len(missing_q1_word))
        fs1.set_value(index, 'q2word_not_in_vocab_length', len(missing_q2_word))

        #Main work starts from here
        q1_vector, q2_vector = get_vector(q1_word, q2_word, Quora_word2vec)
        Questions_In_Vocab(fs1, row, index, q1_vector, q2_vector, q1_word, q2_word)
        Missing_Words_In_Question_In_Vocab(fs1, missing_q1_word, missing_q2_word, index)
        Distance_Functions(Quora_word2vec, fs1, index, q1_vector, q2_vector, row['question1_tokens'],row['question2_tokens'])

    #Write into file here
    Normalized_Word_Mover_Distance(fs1,Quora_word2vec)
    Write_to_csv(fs1,Features,Missing_Features)



def get_vector(q1_word,q2_word,model):

    q1_vector_word=[]
    q2_vector_word=[]



    if len(q1_word)==0 and len(q2_word)>0: #if q1 not found in vocabulary

        for word in q2_word:

            q2_vector_word.append(model[word])  # Vector representation of each word in q1


        q2_vector = np.array(np.sum(q2_vector_word, axis=0) / np.sqrt((np.sum(q2_vector_word, axis=0) ** 2).sum()))
        q1_vector = np.zeros(shape=q2_vector.shape)



    if len(q2_word)==0 and len(q1_word)>0: #if q2 not found in vocabulary

        for word in q1_word:

            q1_vector_word.append(model[word])  # Vector representation of each word in q2

        q1_vector = np.array(np.sum(q1_vector_word, axis=0) / np.sqrt((np.sum(q1_vector_word, axis=0) ** 2).sum()))
        q2_vector = np.zeros(shape=q1_vector.shape)

    if len(q1_word)==0 and len(q2_word)==0:


        q1_vector=np.zeros(shape=(300,1))
        q2_vector=np.zeros(shape=q1_vector.shape)


    if len(q1_word)>0 and len(q2_word)>0:  #If both are found in vocabulary

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

    return q1_vector,q2_vector

def Distance_Functions(model,fs1,index,q1_vector,q2_vector,q1,q2):

    try:
        fs1.set_value(index,'euclidean',Euclidean_Distance(q1_vector,q2_vector))

    except Exception as e :

        print e
        fs1.set_value(index, 'euclidean', 0)

    try:
        fs1.set_value(index,'manhattan',Manhattan(q1_vector,q2_vector))
    except Exception as e :
        print e
        fs1.set_value(index, 'manhattan', 0)
    try:
        fs1.set_value(index,'jaccard',Jaccard(q1_vector,q2_vector))
    except Exception as e :
        print e
        fs1.set_value(index, 'jaccard', 0)
    try:
        fs1.set_value(index,'cossimilarity',Cosine_Similarity(q2_vector,q2_vector))
    except Exception as e :
        print e
        fs1.set_value(index, 'cossimilarity', 0)
    try:
        fs1.set_value(index,'canberra',Canberra(q1_vector,q2_vector))
    except Exception as e :

        print e
        fs1.set_value(index, 'canberra', 0)
    try:
        fs1.set_value(index,'braycurtis',Braycurtis(q1_vector,q2_vector))
    except Exception as e :
        print e
        fs1.set_value(index, 'braycurtis', 0)
    try:
        fs1.set_value(index,'minkowski',Minkowksi(q1_vector,q2_vector,3))
    except Exception as e :
        print e
        fs1.set_value(index, 'minkowski', 0)
    try:
        fs1.set_value(index,'wordmover',Word_Mover_Distance(q1,q2,model))
    except Exception as e :
        print e
        fs1.set_value(index, 'wordmover', 0)
    try:
        fs1.set_value(index, 'kurtosisq1', Kurtosis_q1_vector(q1_vector))
    except Exception as e :
        print e
        fs1.set_value(index, 'kurtosisq1', 0)
    try:
        fs1.set_value(index, 'kurtosisq2', Kurtosis_q2_vector(q2_vector))
    except Exception as e :
        print e
        fs1.set_value(index, 'kurtosisq2', 0)
    try:
        fs1.set_value(index, 'skewq1', Skew_q1_vector(q1_vector))
    except Exception as e :
        print e
        fs1.set_value(index, 'skewq1', 0)
    try:
        fs1.set_value(index, 'skewq2', Skew_q2_vector(q2_vector))
    except Exception as e :
        print e
        fs1.set_value(index, 'skewq2', 0)




def Write_to_csv(fs1,Features,Missing_Features):

    for i in range(0,300):

        Features.append('q1_vector_'+str(i))

    for i in range(0,300):

        Features.append('q2_vector_'+str(i))

    Features.append('is_duplicate')

    #Word 2 vec features

    fs1.to_csv("Feature_set3.csv",columns=Features)

    #Missing words in word2vec-----> for analysis

    fs1.to_csv("missing_words_word2vec.csv",columns=Missing_Features)



def Missing_Words_In_Question_In_Vocab(fs1,missing_q1_word,missing_q2_word,count_found):


        if len(missing_q1_word)>1:

            fs1.set_value(count_found, 'q1_words_not_in_vocab', ",".join(missing_q1_word))

        elif len(missing_q1_word)==1:

            fs1.set_value(count_found, 'q1_words_not_in_vocab', missing_q1_word[0])

        elif len(missing_q1_word)==0:

            fs1.set_value(count_found, 'q1_words_not_in_vocab', 0)

        if len(missing_q2_word)>1:

            fs1.set_value(count_found, 'q2_words_not_in_vocab', ",".join(missing_q2_word))

        elif len(missing_q2_word) == 1:

            fs1.set_value(count_found, 'q2_words_not_in_vocab', missing_q2_word[0])

        elif len(missing_q2_word) == 0:

            fs1.set_value(count_found, 'q2_words_not_in_vocab', 0)


def Questions_In_Vocab(fs1,row,count_found,q1_vector,q2_vector,q1_word,q2_word):





    for i in range(0,q1_vector.shape[1]):



            fs1.set_value(count_found,'q1_vector_'+str(i),q1_vector[:,i])

    for i in range(0,q1_vector.shape[1]):

            fs1.set_value(count_found,'q2_vector_'+str(i), q2_vector[:,i])

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




        #http://nbviewer.jupyter.org/github/vene/vene.github.io/blob/pelican/content/blog/word-movers-distance-in-python.ipynb

#-----------------------------------------Features of vectors------------------------------------------------------------------#-


def Manhattan(q1_vector,q2_vector):


    return spatial.distance.cityblock(q1_vector,q2_vector)

def Cosine_Similarity(q1_vector,q2_vector):

    return spatial.distance.cosine(q1_vector,q2_vector)

def Euclidean_Distance(q1_vector,q2_vector):


    return spatial.distance.euclidean(q1_vector,q2_vector)

def Minkowksi(q1_vector,q2_vector,p):

    return spatial.distance.minkowski(q1_vector,q2_vector,p)

def Canberra(q1_vector,q2_vector):

    return spatial.distance.canberra(q1_vector,q2_vector)

def Braycurtis(q1_vector,q2_vector):

    return spatial.distance.braycurtis(q1_vector,q2_vector)

def Jaccard(q1_vector,q2_vector):

    return spatial.distance.jaccard(q1_vector,q2_vector)

def Kurtosis_q1_vector(q1_vector):

    kurt_q1 = stats.kurtosis(q1_vector.reshape(300,1))[0]
    return kurt_q1

def Kurtosis_q2_vector(q2_vector):

    kurt_q2 =  stats.kurtosis(q2_vector.reshape(300,1))[0]
    return kurt_q2

def Skew_q1_vector(q1_vector):

    skew_q1= stats.skew(q1_vector.reshape(300,1))[0]
    return skew_q1

def Skew_q2_vector(q2_vector):

    skew_q2 = stats.skew(q2_vector.reshape(300,1))[0]
    return skew_q2

def Word_Mover_Distance(q1,q2,model):

    return model.wmdistance(q1, q2)

def Normalized_Word_Mover_Distance(fs1,model):

    model.init_sims(replace=True)
    temp = np.zeros(shape=fs1['question1_tokens'].shape[0])

    for index,row in fs1.iterrows():

            temp=model.wmdistance(fs1.loc[index,'question1_tokens'],fs1.loc[index,'question2_tokens'])
            fs1.set_value(index,'normwordmover',temp)




def Main():
    import time
    start_time = time.time()
    file='train.csv'
    corpus_raw=Data_Cleaning(file)
    words_to_vectors(corpus_raw)
    print("--- %s seconds ---" % (time.time() - start_time))

Main()
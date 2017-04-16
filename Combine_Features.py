#@description: Here we are combining all the three set of features which we have created


import pandas as pd
import numpy as np

def Combine_Features(feature1,feature2,feature3):



    fs1=pd.read_csv(feature1)
    fs2 = pd.read_csv(feature2)
    fs3= pd.read_csv(feature3)

    fs3[np.isnan(fs3)] = 0
    fs3[np.isinf(fs3)] = 0


    fs1_fs2=pd.concat([fs1,fs2],axis=1)

    data = pd.concat([fs1_fs2,fs3],axis=1)

    data = data.loc[:, ~data.columns.duplicated()]


    headers=list(data)
    data.to_csv("Feature_all.csv",columns=headers)


def Main():
    import time
    start_time = time.time()
    feature1= "Feature_set1.csv"    #Basic features
    feature2= "Feature_set2.csv"    #Levenshtein distances
    feature3 = "Feature_set3.csv"  #Word2Vec features

    Combine_Features(feature1,feature2,feature3)
    print("--- %s seconds ---" % (time.time() - start_time))

Main()
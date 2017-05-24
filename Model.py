import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import xgboost as xg
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import  GridSearchCV
import cPickle
import sys 

def Xgb_Boost_Model(file):

    PICKLE_FILE_PATH="XGBOOST_TRAIN"+".pkl"
    dataframe=pd.read_csv(file)
    dataframe= dataframe.replace(np.nan,-5555555)
    dataframe= dataframe.replace(np.inf,-5555555)
    data = np.array(dataframe)
    target=data[:,-1]
    Features=data[:,1:628]
    		
    #Hyper-tuned Parameters of xgboost

    xg_boost=xg.XGBClassifier(learning_rate=0.01,
                              n_estimators=251, max_depth=9,
                              min_child_weight=1, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
                              objective='binary:logistic', nthread=4, scale_pos_weight=1,
                              seed=27)

    xg_boost.fit(Features,target)

    with open(PICKLE_FILE_PATH,'wb') as pickle_file:

        cPickle.dump(xg_boost,pickle_file)	

def Main():


    print("Programe started....")
    file=sys.argv[1] #File name 
    start_time = time.time()
    Xgb_Boost_Model(file)
    print("--- %s seconds ---" % (time.time() - start_time))

Main()

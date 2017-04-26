import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import xgboost as xg
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.manifold import t_sne

def Xgb_Boost_Model():

    dataframe=pd.read_csv('Quora_Engineered_Features.csv')
    dataframe.replace(np.nan,-5555555)
    dataframe.replace(np.inf,-5555555)
    data = np.array(dataframe)
    target=data[:,-1]
    Features=data[:,2:628]


    xg_boost=xg.XGBClassifier(learning_rate =0.1,
             n_estimators=251,
             max_depth=10,
             min_child_weight=1,
             gamma=0.5,
             subsample=0.8,
             colsample_bytree=0.8,
             objective= 'binary:logistic',
             nthread=-1,
             scale_pos_weight=1,
             seed=27)

    AUC=[]
    # log=[]
    kf = KFold(n_splits=5)    #----------> 5 fold cross validation
    for train_index, test_index in kf.split(Features):

        X_train, X_test = Features[train_index], Features[test_index]
        y_train, y_test = target[train_index], target[test_index]
        xg_boost.fit(X_train, y_train)
        y_pred = xg_boost.predict(X_test)
        AUC.append(accuracy_score(y_test, y_pred))

    print "Accuracy", np.mean(AUC)*100

    #Cross-Validated-Accuracy 0.753708971283(5-fold cross validation) estimators=100


   

def Main():

    import time
    print("Programe started....")
    start_time = time.time()
    Xgb_Boost_Model()
    print("--- %s seconds ---" % (time.time() - start_time))

Main()

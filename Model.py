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

def read_file(file):

    # filter(function, iterable) is equivalent to[item for item in iterable if function(item)]


    # dataframe=pd.read_csv(file)
    #
    # cols = list(dataframe)
    # col=[]
    # exclude=['Unnamed: 0',"Unnamed: 0.1",'id','is_duplicate']
    # col=filter(lambda index: index not  in exclude,cols )
    #
    # data = np.array(dataframe.values)
    # target = np.array(data[:, 11])
    #
    # All_Features = np.array(dataframe[col].copy())  #All set of features
    #
    #
    # w2v_Features = All_Features[:, 15:All_Features.shape[1]] #only wordvector features
    #
    # vec_Features = All_Features[:,28:All_Features.shape[1]] #only word vectors
    #
    # # stratified sample
    #
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    # All_Features, target, test_size=0.1,stratify=target, random_state = 0)
    #
    # y_test=y_test.reshape(y_test.shape[0],1)
    # stratified_data = np.concatenate((X_test,y_test),axis=1)
    # np.savetxt('stratifiedsample.csv',stratified_data)
    # exit(0)
    data = np.loadtxt('stratifiedsample.csv')
    All_Features=data[:,0:628]
    target=data[:,-1]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    All_Features, target, test_size=0.1,random_state = 0)
    return X_train,X_test,y_train,y_test

def Logistic_Regression_Model(X_train,X_test,y_train,y_test):

    #I am using stratified sample as the data is huge and takes alot of time to train
    clf=LogisticRegression()
    clf.c=1.0
    clf.penalty='l1'
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print "Accuracy: " + str(accuracy_score(y_test,y_pred)*100)

    #---Accuracy 74.5%------# -- > This accuracy is on complete training set
    #Hyper parameters can be tuned to get much better accuracy



def Xgb_Boost_Model(data,target): #Needs to be implemented

    return "b"

def Visualization(data):


    # All features mapped to 2D

    dataframe = pd.read_csv(data,nrows=100000)
    clf = PCA(n_components=2)
    cols = list(dataframe)
    col = []
    exclude = ['Unnamed: 0', "Unnamed: 0.1", 'id', 'is_duplicate']
    col = filter(lambda index: index not in exclude, cols)

    data = np.array(dataframe.values)
    target = np.array(data[:, 11])

    Features = np.array(dataframe[col].copy())

    Vector_Features = Features[:,15:len(Features)]


    Transformed_2D = clf.fit(Vector_Features, target).transform(Vector_Features)
    dataframe['colors'] = dataframe.apply(lambda row: 'red' if row['is_duplicate'] == 0 else 'blue', axis=1)

    plt.scatter(Transformed_2D[:, 0], Transformed_2D[:, 1], color=dataframe['colors'])

    plt.legend(loc='lower left', ncol=3, fontsize=7)
    plt.title("All features projected on 2 dimension")
    plt.show()









def Main():

    import time
    start_time = time.time()
    file='Feature_all.csv'
    X_train,X_test,y_train,y_test=read_file(file)
    Logistic_Regression_Model(X_train,X_test,y_train,y_test)
    #Xgb_Boost_Model()
    #Visualization(file)


    print("--- %s seconds ---" % (time.time() - start_time))

Main()

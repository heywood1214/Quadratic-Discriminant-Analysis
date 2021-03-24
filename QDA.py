import pandas as pd
import numpy as np 
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix

warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer 

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names) # load data into data frame using panda

df.describe #look at each row 
df.info() # evaluate each variable
df.head # first 6 rows of data

df['target']=pd.Series(data.target)

#need to find the mu, sigma, and pi to find the determinant 

#break the data into 2 classes based on class 1 and class 0, then find the according mu, sigma, and pi using 30 parameters

print(df.groupby('target').count())#2 classes based on 0,1 
print(df.groupby('target').count().shape)# 2 classes, and 30 parameters
K,p = df.groupby('target').count().shape

#30 mu respectively for 2 classes
print(np.split(df.groupby('target').mean().values,K)) 
mu = np.split(df.groupby('target').mean().values,K)

#2 classes, 30 x 30 matrix
print(np.split(df.groupby('target').cov().values,K)) 
sigma = np.split(df.groupby('target').cov().values,K)

#pi : nk/n
print(df.groupby('target').size()) # 212 class 0 | 357 class 1
print(len(df)) #569 total rows 
print(np.split(np.array(df.groupby('target').size()/len(df)),K)) # class 1/total rows & class 2/total rows  
pi = np.split(np.array(df.groupby('target').size()/len(df)),K)

#get the determinant for later
def determinant_k(x,mu_k, sigma, pi_k):
    #return ((-1/2)*(x-mu_k).transpose@ np.linalg.inv(sigma)@ (x-mu_k))+(-1/2)*np.log(np.linalg.det(sigma)+np.log(pi_k))
    return ((-1/2)*(x - mu_k).T @ np.linalg.inv(sigma) @ (x - mu_k)) + (-1/2)*np.log(np.linalg.det(sigma)) + np.log(pi_k)



#Prediction using the classifer abov
def target_prediction(X):
    determinant=[]

    for target in range(K):
        #make one dimensional arrays to find (x-mu_k) | get the list we created above to get sigma, and pi 
        target = determinant_k(X.to_numpy().reshape(p,1),mu[target].reshape(p,1),sigma[target],pi[target]) 
        determinant.append(target)

    return np.argmax(determinant)

#error rate
def error_rate(dataframe): 
    return np.where(dataframe['target']==dataframe['prediction'],0,1).sum()/len(dataframe)


#get the error rate 

df['prediction'] = df.drop('target',axis=1).apply(target_prediction, axis=1)
error_rate(df)



class kFoldCV: 

    def __init__(self):
        pass

    def crossValidateSplit(self, dataset):

        data_split = list()
        data_copy = list(dataset)
        foldsize = int(len(dataset)/5)
        for _ in range(5):
            fold = list()
            while len(fold) < foldsize:
                index = randrange(len(data_copy))
                fold.append(datacopy.pop(index))
            data_split.append(fold)
        return data_split

    def kfold_evaluate(self, dataset):
        target_prediction = target_prediction()
        scores = list()
        for fold in folds: 
            training = list(folds)
            training.remove(fold)
            training = sum(training, [])
            training= list()
        for row in fold: 
            rowCopy = list(row)
            testing.append(rowCopy)

            trainLabels = [row[-1] for row in training]
            trainSet = [train[:-1] for train in training]
            target_prediction(trainSet,trainLabels)
            
            actual = [row[-1] for row in testing]
            testSet = [test[:-1] for test in testing]
            
            predicted = target_prediction(testSet)
            accuracy = printMetrics(actual, predicted)
            scores.append(accuracy)
            print(scores)
            
        
exit 

'''
kfcv = kFoldCV()
kfcv.kfold_evaluate(df)
'''

#use built in QDA from sklearn
from sklearn.datasets import load_breast_cancer 
data = load_breast_cancer()
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

df_sklearn = pd.DataFrame(data.data, columns=data.feature_names)
df_sklearn['target']=pd.Series(data.target)
x_var = df_sklearn.drop(('target'),axis=1)
y_var = df_sklearn['target']

qda = QuadraticDiscriminantAnalysis()
qda_fit = qda.fit(x_var,y_var)

prediction = qda_fit.predict(x_var)
print(prediction)

print(1-qda.score(x_var,y_var))


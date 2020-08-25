#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 01:36:07 2020

This is the skeleton of the supervised machine learning model we want to apply
to Britta's sequence data. It consists of libraries from scikit learn and 
epitope predict



@author: jcs
"""
#Import of required libraries 
import os, sys, math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import epitopepredict as ep
from sklearn import metrics

from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import csv
from collections import Counter

#The following functions outline different encoding methods
#Note that as of now epitopepredict does not function on Windows OS
#Encoders are from http://dmnfarrell.github.io/bioinformatics/mhclearning

#One hot encode matrix 

def one_hot_encode(seq):
    codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    o = list(set(codes) - set(seq))
    s = pd.DataFrame(list(seq))    
    x = pd.DataFrame(np.zeros((len(seq),len(o)),dtype=int),columns=o)    
    a = s[0].str.get_dummies(sep=',')
    a = a.join(x)
    a = a.sort_index(axis=1)
    e = a.values.flatten()
    return e

#NLF matrix 

def nlf_encode(seq):    
    nlf = pd.read_csv('NLF.csv',index_col=0)
    x = pd.DataFrame([nlf[i] for i in seq]).reset_index(drop=True)  
    e = x.values.flatten()
    return e

#Blosum 62 matrix 

def blosum_encode(seq):
    
    blosum = ep.blosum62
    s=list(seq)
    x = pd.DataFrame([blosum[i] for i in seq]).reset_index(drop=True)
    e = x.values.flatten()    
    return e

#In this portion of the code we will import the data needed 

df = pd.read_csv("for_JC_new.csv") 


#Next we need to define the Area Under the Curve using scikit 
#This will tell us the machine's ability to 'guess' 


def auc_score(true,sc,cutoff=None):

    if cutoff!=None:
        true = (true<=cutoff).astype(int)
        sc = (sc<=cutoff).astype(int) 
    fpr, tpr, thresholds = metrics.roc_curve(true, sc, pos_label=1)
    r = metrics.auc(fpr, tpr)
    return  r

#Next we need to split the data into the training and the testing sets
#And normalize the data

def data_splitter(encoder, test_percent=0.3):
    X = df.Sequence.apply(lambda x: pd.Series(encoder(x)),1) #place holder
    y = df.dG #Use this for the regressor 
    #y = df.stable #Use this for the classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_percent)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)       
    return X_train, X_test, y_train, y_test
    
#Its time to create the predictor, which we will have a few options
#Such as using a classifier vs a regression model, the type of encoder
#And adjusting specific hyperparameters 

def Classifier_MLP(encoder, test_percent=0.3):
    X_train, X_test, y_train, y_test = data_splitter(encoder, test_percent=test_percent)
    mlp = MLPClassifier(hidden_layer_sizes=(20,20,20),activation='relu', solver='adam', max_iter=5000)
    mlp.fit(X_train, y_train)
    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)
    return predict_train, predict_test, y_train, y_test

#With the classifier generated and tested we can evaluate its accuracy

def did_i_do_good(encoder, test_percent=0.3):
    
    predict_train, predict_test, y_train, y_test = Classifier_MLP(encoder, test_percent=test_percent)
    print(confusion_matrix(y_test,predict_test))
    print(classification_report(y_test,predict_test))


#The next model we can use is the regressor model to predict the stability of a sequence

def Regressor_MLP(encoder, test_percent=.3):
    mlp = MLPRegressor(hidden_layer_sizes=(20,20,20), alpha=0.01, max_iter=500,
                        activation='relu', solver='lbfgs', random_state=2)
    X_train, X_test, y_train, y_test = data_splitter(encoder, test_percent=test_percent)
    print(len(X_test))
    print(len(y_test))
    mlp.fit(X_train, y_train)
    sc = mlp.predict(X_test)
    auc = auc_score(y_test,sc,cutoff=-33.3)
    x=pd.DataFrame(np.column_stack([y_test,sc]),columns=['test','predicted'])
    ax = sns.regplot(y_test, sc, fit_reg=True,scatter_kws={"s":100},line_kws={'color':'black'})
    plt.text(-50,-30,'auc=%s' %round(auc,2))
    ax.set(xlabel='Actual dG', ylabel='Predicted dG')
    plt.savefig('encoder_'+str(encoder)+'.png')
    print(); print(metrics.r2_score(y_test,sc))
    print(); print(metrics.mean_absolute_error(y_test,sc))
    return mlp 


plt.clf()
Regressor_MLP(blosum_encode)







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 00:42:25 2020

@author: sh
"""


#%%
from __future__ import print_function

import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

def split_data():
    path = r'/home/sh/anaconda3/fraud/ieee-fraud-detection/'
    X = pd.read_csv(path+'train_graph_node.csv', index_col='Unnamed: 0')
    idx = X['uid'].unique()
    label = []
    for i in idx:
        q = X[X['uid']==i]
        if np.mean(q['isFraud']) == 0:
            label.append(0)
        elif np.mean(q['isFraud']) == 1:
            label.append(1)
        else:
            label.append(2)
    train_label, test_label, y_train, y_test = train_test_split(idx, label, test_size=0.25, stratify=label, random_state=0)
    X_train = pd.DataFrame(columns=X.columns)
    X_test = pd.DataFrame(columns=X.columns)
    for i in idx:
        q = X[X['uid']==i]
        if i in train_label:
            X_train = X_train.append(q)
        else:
            X_test = X_test.append(q)
    y_train = X_train[['uid', 'isFraud', 'time']]
    y_test = X_test[['uid', 'isFraud', 'time']]
    del X_train['isFraud']
    del X_test['isFruad']
    X_train.to_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/X_train_graph_node.csv')
    X_test.to_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/X_test_graph_node.csv')
    y_train.to_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/y_train_graph_node.csv')
    y_test.to_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/y_test_graph_node.csv')

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def edge_create(var, time):
    X_train = pd.read_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/X_train_graph_node.csv', index_col='Unnamed: 0')
    X_test = pd.read_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/X_test_graph_node.csv', index_col='Unnamed: 0')
    X_train = X_train[['uid', var]]
    X_test = X_test[['uid', var]]

    idx_train = X_train['uid'].unique()
    idx_test = X_test['uid'].unique()

    X_train.drop_duplicates(inplace=True)
    X_test.drop_duplicates(inplace=True)
    
    n = len(idx_train)
    train_neign = np.array([0]*(n**2)).reshape(n, n)
    for v in tqdm(range(n)):
        train_neign[v] = (X_train.iloc[v, 1]==X_train[var])
    for j in range(n):
        train_neign[j, j] = 0
        
    n = len(X_test)
    test_neign = np.array([0]**(n**2)).reshape(n, n)
    for v in tqdm(range(n)):
        test_neign[v] = (X_test.iloc[v, 1]==X_test[var])
    for j in range(n):
        test_neign[j ,j] = 0
    
    train_neign = sp.csr_matrix(train_neign)
    test_neign = sp.csr_matrix(test_neign)
    
    return train_neign, test_neign, idx_train, idx_test

def load_features(var, time):
    X_train = pd.read_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/X_train_graph_node.csv', index_col='Unnamed: 0')
    X_test = pd.read_csv(r'/home/sh/anaconda3/fruad/ieee-fraud-detection/X_test_graph_node.csv', index_col='Unnamed: 0')
    X_train = X_train[X_train['time']==time]
    X_test = X_test[X_test['time']==time]
    del X_train[var]
    del X_test[var]
    X_train = np.matrix(X_train.values)
    X_test = np.matrix(X_test.values)
    X_train_features = preprocess_features(X_train)
    X_test_features = preprocess_features(X_test)
    return X_train_features, X_test_features

def load_label(time):
    y_train = pd.read_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/y_train_graph_node.csv', index_col='Unnamed: 0')
    y_test = pd.read_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/y_test_graph_node.csv', index_col='Unnamed: 0')
    y_train = y_train[y_train['time']==time]['isFraud'].values
    y_test = y_test[y_test['time']==time]['isFraud'].values
    train_label = np.array([0]*(2*len(y_train))).reshape((-1, 2))
    test_label = np.array([0]*(2*len(y_test))).reshape((-1, 2))
    for i in y_train:
        if i==0:
            train_label[i][0] = 1
        else:
            train_label[i][1] = 1
    for i in y_test:
        if i==0:
            test_label[i][0] = 1
        else:
            test_label[i][1] = 1
    
    return train_label, test_label



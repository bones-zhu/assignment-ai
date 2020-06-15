#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 00:42:25 2020

@author: sh
"""



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
    X = pd.read_csv(path+'train_graph_node.csv')
    idx = X['uid'].unique()
    label = []
    for i in tqdm(idx):
        q = X[X['uid']==i]
        if np.mean(q['isFraud']) == 0:
            label.append(0)
        elif np.mean(q['isFraud']) == 1:
            label.append(1)
        else:
            label.append(2)
    train_label, test_label, y_train, y_test = train_test_split(idx, label, test_size=0.25, stratify=label, random_state=0)
    train_label, val_label, y_train, y_val = train_test_split(train_label, y_train, test_size=0.25, stratify=y_train, random_state=0)
    del X
    n = len(idx)
    mask_train = np.array([0]*n)
    mask_val = np.array([0]*n)
    mask_test = np.array([0]*n)
    for i in range(n):
        if idx[i] in train_label:
            mask_train[i] = 1
        elif idx[i] in val_label:
            mask_val[i] = 1
        else:
            mask_test[i] = 1
    np.save('/home/sh/anaconda3/fraud/ieee-fraud-detection/mask_train', mask_train)
    np.save('/home/sh/anaconda3/fraud/ieee-fraud-detection/mask_val', mask_val)
    np.save('/home/sh/anaconda3/fraud/ieee-fraud-detection/mask_test', mask_test)
    #X_train = pd.DataFrame(columns=X.columns)
    #X_test = pd.DataFrame(columns=X.columns)
    #for i in idx:
    #    q = X[X['uid']==i]
    #    if i in tqdm(train_label):
    #        X_train = X_train.append(q)
    #    else:
    #        X_test = X_test.append(q)
    #y_train = X_train[['uid', 'isFraud', 'time']]
    #y_test = X_test[['uid', 'isFraud', 'time']]
    #del X_train['isFraud']
    #del X_test['isFruad']
    #X_train.to_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/X_train_graph_node.csv')
    #X_test.to_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/X_test_graph_node.csv')
    #y_train.to_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/y_train_graph_node.csv')
    #y_test.to_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/y_test_graph_node.csv')

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    for t in tqdm(range(27)):
        rowsum = np.array(features[t].sum(1)) 
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        features[t] = r_mat_inv.dot(features[t])
    return features

def edge_create(var):
    X_train = pd.read_csv('/home/sh/anaconda3/fraud/ieee-fraud-detection/train_graph_node.csv')
    X_train = X_train[['uid', var]]
    idx = X_train['uid'].unique()
    idx = sorted(idx)
    n = len(idx)
    c = dict()
    for i in tqdm(range(len(idx))):
        c[idx[i]] = i
    for i in tqdm(X_train.index):
        X_train.loc[i, 'uid'] = c[X_train.loc[i, 'uid']]
    del idx
    card2 = X_train[var].unique()
    indx = []
    indy = []
    for i in tqdm(range(len(card2))):
        q = X_train[X_train[var]==i]['uid'].unique()
        for j in range(len(q)-1):
            for t in range(j+1, len(q)):
                indx.append(q[j])
                indy.append(q[t])
    n = len(X_train['uid'].unique())
    n1 = len(indx)
    del X_train, var, card2
    indx = np.array(indx)
    indy = np.array(indy)
    edge = sp.coo_matrix((np.array([1]*n1), (indx, indy)), shape=(n, n))
    del indx, indy
    edge = sp.csr_matrix(edge)
    sp.save_npz('/home/sh/anaconda3/fraud/ieee-fraud-detection/edge.npz', edge)
    return edge





def load_features(var):
    X_train = pd.read_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/train_graph_node.csv')
    N_train = len(X_train['uid'].unique())
    F = X_train.shape[1] - 5
    del X_train[var]
    c = dict()
    idx = X_train['uid'].unique()
    col = list(X_train.columns[2:-2])
    for i in tqdm(range(len(idx))):
        c[idx[i]] = i
    train_matrix = np.zeros((27, N_train, F))
    for t in tqdm(range(27)):
        q = X_train[X_train['time']==t]
        for j in q.index:
            axis = q.loc[j, 'uid']
            train_matrix[t, c[axis]] = q.loc[j, col]
    del X_train, c, idx, col
    train_matrix = preprocess_features(train_matrix)
    return train_matrix

def load_y():
    X_train = pd.read_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/train_graph_node.csv')[['isFraud', 'uid', 'time']]
    idx = sorted(X_train['uid'].unique())
    n = len(idx)
    c = dict()
    for i in range(n):
        c[idx[i]] = i
    for i in tqdm(X_train.index):
        X_train.loc[i, 'uid'] = c[X_train.loc[i, 'uid']]
    y = np.zeros((27, n))
    for t in tqdm(range(27)):
        q = X_train[X_train['time']==t]
        for i in q.index:
            y[t, int(q.loc[i, 'uid'])] = int(q.loc[i, 'isFraud']) + 1
    del X_train
    y = y.T
    for i in tqdm(range(n)):
        for t in range(1, 27):
            if y[i, t] != 0:
                continue
            y[i, t] = y[i, t-1]
    for i in tqdm(range(n)):
        for t in range(26, -1, -1):
            if y[i, t] != 0:
                continue
            y[i, t] = y[i, t+1]
    y = y.T
    y -= 1
    y_label = np.zeros((27, n, 2))
    for i in tqdm(range(27)):
        for j in range(n):
            y_label[i, j, int(y[i, j])] = 1
    #np.save('/home/sh/anaconda3/fraud/ieee-fraud-detection/y_label', y_label)
    return y_label
                

        

def load_label():
    mask_train = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/mask_train.npy')
    mask_val = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/mask_val.npy')
    mask_test = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/mask_test.npy')
    X_train = pd.read_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/train_graph_node.csv')[['isFraud', 'uid', 'time']]
    #y_train = pd.read_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/y_train_graph_node.csv', index_col='Unnamed: 0')
    #y_test = pd.read_csv(r'/home/sh/anaconda3/fraud/ieee-fraud-detection/y_test_graph_node.csv', index_col='Unnamed: 0')
    y_label = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/y_label.npy') # (27, n, 2)
    mask_train = mask_train.astype(bool)
    mask_val = mask_val.astype(bool)
    mask_test = mask_test.astype(bool)
    n = y_label.shape[1]
    y_train = np.zeros((27, n, 2))
    y_val = np.zeros((27, n, 2))
    y_test = np.zeros((27, n, 2))
    
    y_train[:, mask_train] = y_label[:, mask_train]
    y_val[:, mask_val] = y_label[:, mask_val]
    y_test[:, mask_test] = y_label[:, mask_test]
    
    return y_train, y_val, y_test



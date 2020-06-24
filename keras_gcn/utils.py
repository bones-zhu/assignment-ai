#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:41:17 2020

@author: sh
"""


from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from tqdm import tqdm

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c:np.identity(len(classes))[i,:] for i,c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data():
    features = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/X.npy')
    x = list()
    for i in tqdm(range(27)):
        x.append(sp.csr_matrix(features[i]).todense())
    del features
    adj = sp.load_npz('/home/sh/anaconda3/fraud/ieee-fraud-detection/edge.npz')
    adj = adj.astype(np.int16)
    adj = sp.coo_matrix(adj)
    return x, adj

def normalize_adj(adj, symmetric=True): #用度矩阵标准化连接矩阵D(-0.5)@A@D(-0.5)
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    a_norm = a_norm.astype(np.float32)
        
    return a_norm

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0]) #邻接矩阵加自连接
    adj = normalize_adj(adj, symmetric)
    return adj

def get_splits():
    y_label = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/y_label.npy')

    idx_train = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/idx_train.npy')
    idx_test = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/idx_test.npy')
    idx_val = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/idx_val.npy')
    return idx_train, idx_val, idx_test, y_label

def mask():
    mask_train = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/mask_train.npy')
    mask_val = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/mask_val.npy')
    mask_test = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/mask_test.npy')
    train = list()
    val = list()
    test = list()
    for i in range(mask_train.shape[0]):
        train.append(sp.csr_matrix(mask_train[i]))
        val.append(sp.csr_matrix(mask_val[i]))
        test.append(sp.csr_matrix(mask_test[i]))
    del mask_train, mask_val, mask_test
    return train, val, test

def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))

def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1)), np.argmax(preds, 1))

def evalute_preds(preds, labels, indices):
    
    split_loss = list()
    split_acc = list()
    
    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))
        
    return split_loss, split_acc

def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian

def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2
        
    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    scaled_laplacian = scaled_laplacian.astype(np.float16)
    return scaled_laplacian

def chebyshev_polynomial(X, k): #   切比雪夫多项式近似，计算k阶的切比雪夫近似矩阵
    """
    Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices.
    """
    print('Calculating Chebyshev polynomials up to order {}...'.format(k))
    
    T_k = list()
    T_k.append(sp.eye(X.shape[0]).astype(np.float16).tocsr())
    T_k.append(X)
    
    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return (2 * X_.dot(T_k_minus_one) - T_k_minus_two).astype(np.float16)
    
    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))
        
    return T_k

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
    
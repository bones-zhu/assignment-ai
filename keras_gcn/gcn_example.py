#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:32:26 2020

@author: sh
"""


from __future__ import print_function

from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import gc

from keras_gcn.layers.graph import GraphConvolution
from keras_gcn.utils import *

import time

#Define parameters
FILTER = 'localpool'
MAX_DEGREE = 2 #maximum polynomial degree
SYM_NORM = True
NB_EPOCH = 200
PATIENCE = 10 #early stopping patience
lr = 0.01
dropout_rate = 0.5
l2_reg = 5e-4

#Get data
X, A = load_data()
idx_train, idx_val, idx_test, train_mask = get_splits()
N = X[0].shape[0]
T = len(X)
F = X[0].shape[1]
F_1 = 16
F_2 = 2
#%%
#Normalize X
for i in range(T):
    X[i] /= X[i].sum(1).reshape(-1, 1)
    X[i][np.isnan(X[i])] = 0
    X[i][np.isinf(X[i])] = 0
gc.collect()
#%%
if FILTER == 'localpool':
    """
    Local pooling filters (see 'renormalization trick' in Kipf & Welling, aeXiv 2016)
    """
    print("Using local pooling filters...")
    A_ = preprocess_adj(A, SYM_NORM) #拉普拉斯矩阵
    support = 1
    graph = []
    for i in range(T):
        graph.append(X[i])
        graph.append(A_)
#    G = [Input(shape=(None, None), sparse=True)]
#    G = [Input(shape=(None,))]
    
elif FILTER == 'chebyshev':
    """
    CEebyshev polynomial basis filters (Defferard et al., NIPS 2016)
    """
    print('Using Chebyshev polynomial basis filters...')
    L = normalized_laplacian(A, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    support = MAX_DEGREE + 1
    graph = []
    for i in range(T):
        graph.append(X[i])
        graph.append(T_k)
#    G = [Input(shape=(None, None), sparse=True) for _ in range(support)]
#    G = [Input(shape=(None,)) for _ in range(support)]

else:
    raise Exception('Invalid filter type.')
gc.collect()
#%%
y_train, y_val, y_test = mask()
inputs = []
outputs = []
for i in range(T):
    s1 = 'output' + str(2*i)
    s2 = 'output' + str(2*i+1)
    X_in = Input(shape=(F,))
    inputs.append(X_in)
    if FILTER == 'localpool':
        G = [Input(shape=(None,))]
    else:
        G = [Input(shape=(None,)) for _ in range(support)]
    inputs.extend(G)
    #Define model architecture
    H = Dropout(dropout_rate)(X_in)
    H = GraphConvolution(F_1, support, activation='relu', kernel_regularizer=l2(l2_reg))([H]+G)
    H = Dropout(dropout_rate)(H)
    Y = GraphConvolution(F_2, support, activation='softmax')([H]+G)
    outputs.append(Y)    


#Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999



Y_train_ = dict()
loss_weight = dict()
for i in range(T):
    s = 'graph_convolution_' + str(2*i+1)
    Y_train_[s] = y_train[i]
    loss_weight[s] = 1/27
y_train = Y_train_
del Y_train_
id_train = np.zeros((55017,))
id_train[idx_train] = 1

idx_train, idx_val, idx_test = [id_train]*T, [idx_val]*T, [idx_test]*T
gc.collect()
del id_train
gc.collect()
#%%
#Compile model
model = Model(inputs=inputs, outputs=outputs)

#%%
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), loss_weight=loss_weight, weighted_metrics=['acc'])
#%%
#fit
for epoch in range(1, NB_EPOCH+1):
    #Log wall-clock time
    t = time.time()
    
    #Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y_train, sample_weight=idx_train, 
              batch_size=A.shape[0], epochs=False, verbose=0)

    #Predict on full dataset
    preds = model.predict(graph, batch_size=A.shape[0])
    
    #Train / validation scores
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val], 
                                                   [idx_train, idx_val])
    print('Epoch: {:04d}'.format(epoch), 
          'train_loss={:.4f}'.format(train_val_loss[0]), 
          'train_acc={:.4f}'.format(train_val_acc[0]), 
          'val_loss={:.4f}'.format(train_val_loss[1]), 
          'val_acc={:.4f}'.format(train_val_acc[1]), 
          'time={:.4f}'.format(time.time() - t))
          
    
    #Early stopping
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait += 1
#%%
#Testing
test_loss, test_acc = evalute_preds(preds, [y_test], idx_test)
print('Test set results:', 
      'loss= {:.4f}'.format(test_loss[0]), 
      'accuaracy={:.4f}'.format(test_acc[0]))

    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 00:46:15 2020

@author: sh
"""
#%%
import sys
sys.path.append('/home/sh/anaconda3/fraud')

#%%
from __future__ import division

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import scipy.sparse as sp
from tensorflow.keras import backend as K
from tqdm import tqdm

from keras_gat import GraphAttention
from utils import load_features, load_label
#from keras_gat import GraphAttention
#from keras_gat.utils import edge_create, load_features, load_label

# Read data
var = 'card2'
A = sp.load_npz('/home/sh/anaconda3/fraud/ieee-fraud-detection/edge.npz') #(N, N)
X = load_features(var) # (27, N, F)
Y_train, Y_val, Y_test = load_label() # (27, n, 2)
idx_train = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/mask_train.npy') #(N,)
idx_val = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/mask_val.npy') # (N,)
idx_test = np.load('/home/sh/anaconda3/fraud/ieee-fraud-detection/mask_test.npy') #(N,)
# Parameters
N = X.shape[1]                # Number of nodes in the graph
F = X.shape[2]                # Original feature dimension
n_classes = 2
n_classes = Y_train.shape[-1]  # Number of classes
#n_classes = 30
F_ = 10                        # Output size of first GraphAttention layer
n_attn_heads = 8              # Number of attention heads in first GAT layer
dropout_rate = 0.6            # Dropout rate (between and inside GAT layers)
l2_reg = 5e-4/2               # Factor for l2 regularization
learning_rate = 5e-3          # Learning rate for Adam
epochs = 10000                # Number of training epochs
es_patience = 100             # Patience fot early stopping
#p = []
#%%
for time in tqdm(range(2)):

#A, X, Y_train, Y_val, Y_test, idx_train, idx_val, idx_test = load_data('cora')




# Preprocessing operations
#A = A + np.eye(A.shape[0])  # Add self-loops




# Model definition (as per Section 3.3 of the paper)
    X_in = Input(shape=(F,))
    A_in = Input(shape=(N,))

    dropout1 = Dropout(dropout_rate)(X_in)
    graph_attention_1 = GraphAttention(F_,
                                       attn_heads=n_attn_heads,
                                       attn_heads_reduction='concat',
                                       dropout_rate=dropout_rate,
                                       activation='elu',
                                       kernel_regularizer=l2(l2_reg),
                                       attn_kernel_regularizer=l2(l2_reg))([dropout1, A_in])
    dropout2 = Dropout(dropout_rate)(graph_attention_1)
    graph_attention_2 = GraphAttention(n_classes,
                                       attn_heads=1,
                                       attn_heads_reduction='average',
                                       dropout_rate=dropout_rate,
                                       activation='softmax',
                                       kernel_regularizer=l2(l2_reg),
                                       attn_kernel_regularizer=l2(l2_reg))([dropout2, A_in])

# Build model
    model = Model(inputs=[X_in, A_in], outputs=graph_attention_2)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  weighted_metrics=['acc'])
    #model.summary()
#output_array = model.predict([X_train_features, A_train])
#    weight = model.get_weights()
#    output = []
#    for i in range(n_attn_heads):
#        output.append(X[time] @ weight[4*i] + weight[1+4*i])
#    p.append(np.array(K.mean(K.stack(output), axis=0)))
#p = np.arry(p)

# Callbacks
    es_callback = EarlyStopping(monitor='val_weighted_acc', patience=es_patience)
    tb_callback = TensorBoard(batch_size=N)
    mc_callback = ModelCheckpoint('/home/sh/anaconda3/fraud/best_model'+str(time)+'.h5',
                                  monitor='val_weighted_acc',
                                  save_best_only=True,
                                  save_weights_only=True)



# Train model
    validation_data = ([X[time], A], Y_val[time], idx_val)
    model.fit([X[time], A],
              Y_train[time],
              sample_weight=idx_train,
              epochs=epochs,
              batch_size=N,
              validation_data=validation_data,
              shuffle=False,  # Shuffling data means shuffling the whole graph
              callbacks=[es_callback, tb_callback, mc_callback])

# Load best model
    model.load_weights('/home/sh/anaconda3/fraud/best_model'+str(time)+'.h5')

# Evaluate model
    eval_results = model.evaluate([X[time], A],
                                  Y_test,
                                  sample_weight=idx_test,
                                  batch_size=N,
                                  verbose=0)
    print('time:{}'.format(time))
    print('Done.\n'
          'Test loss: {}\n'
          'Test accuracy: {}'.format(*eval_results))

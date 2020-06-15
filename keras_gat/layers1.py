#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:19:09 2020

@author: sh
"""


from __future__ import absolute_import

from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU
import numpy as np
import tensorflow as tf

class GraphAttention(Layer):
    def __init__(self, F_, attn_heads=1, attn_heads_reduction='concat', 
                 dropout_rate=0.5, activation='relu', use_bias=True, 
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                 attn_kernel_initializer='glorot_uniform', kernel_regularizer=None, 
                 bias_regularizer=None, attn_kernel_regularizer=None, 
                 activity_regularizer=None, kernel_constraint=None, 
                 bias_constraint=None, attn_kernel_constraint=None, **kwargs):
        """
        F_:number of output features (F' in the paper)
        attn_heads: number of attention heads(K in the paper)
        attn_heads_reduction: Eq. 5 and 6 in the paper
        dropout_rate: internal dropout rate
        activation: Eq. 4 in the paper
        use_bias: whether to use bias
        kernel_initializer: W初始化
        bias_initializer: 偏置初始化
        attn_kernel_initializer: K初始化
        kernel_regularizer: W正则化
        bias_regularizer: b正则化
        attn_kernel_regularizer: K正则化
        activity_regularizer: 激活函数正则化
        kernel_constraint: W约束
        bias_constraint: b约束
        attn_kernel_constraint: K约束
        """
        
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
            
        self.F_ = F_
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False
        
        #Populated by build()
        self.kernels = [] #Layer kernels for attention heads
        self.biases = [] #Layer biases for attention heads
        self.attn_kernels = [] #Attention kernels for attention heads
        
        if attn_heads_reduction == 'concat':
            self.output_dim = self.F_ * self.attn_heads #K * F'
        else:
            self.output_dim = self.F_
        
        super(GraphAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]
        
        #initialize weights for each attention head
        for head in range(self.attn_heads):
            #layer kernel
            kernel = self.add_weight(shape=(F, self.F_), 
                                     initializer = self.kernel_initializer, 
                                     regularizer = self.kernel_regularizer, 
                                     constraint = self.kernel_constraint, 
                                     name = 'kernel_{}'.format(head))
            self.kernels.append(kernel) #W in Eq. 1
            
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ), 
                                       initializer = self.bias_initializer, 
                                       regularizer = self.bias_regularizer, 
                                       constraint = self.bias_constraint, 
                                       name = 'bias_{}'.format(head))
                self.biases.append(bias) #b in Eq.1
        
            #Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1), 
                                               initializer = self.attn_kernel_initializer, 
                                               regularizer = self.attn_kernel_regularizer, 
                                               constraint = self.attn_kernel_constraint, 
                                               name = 'attn_kernel_self_{}'.format(head))
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1), 
                                                 initializer = self.attn_kernel_initializer, 
                                                 regularizer = self.attn_kernel_regularizer, 
                                                 constraint = self.attn_kernel_constraint, 
                                                 name = 'attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs]) #a in Eq.2
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + K.transpose(attn_for_neighs)  # (N x N) via broadcasting

            # Add nonlinearty
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - A)
            dense += mask

            # Apply softmax to get attention coefficients
            dense = K.softmax(dense)  # (N x N)

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')

            # Linear combination with neighbors' features
            node_features = K.dot(dropout_attn, dropout_feat)  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        return output


#    def call(self, inputs, alpha=0.2):
#        X = inputs[0] # Node features (t, N, F)
#        A = inputs[1] # Adjacency matrix (N, N)
#        t = X.shape[0]
#        N = X.shape[1]
        
#        outputs = []
#        for head in range(self.attn_heads):
#            kernel = self.kernels[head] # W in the paper (F, F')
#            attention_kernel = self.attn_kernels[head] # Attention kernel a in the paper (2F', 1)
            
            #Compute feature combinations
#            features = K.dot(X, kernel) # (t, N, F')
            
            #Compute feature combinations
#            attn_for_self = K.dot(features, attention_kernel[0]) # (t, N, 1)
#            attn_for_neighs = K.dot(features, attention_kernel[1]) # (t, N, 1)
            
            #Attention head a(Wh_i, Wh_j) = a^T @ [[Wh_i], [Wj_j]]
#            dense = attn_for_self + tf.transpose(attn_for_neighs, perm=[0, 2, 1]) # (t, N, N) via broadcasting
            
            #Add nonlinearty
#            dense = LeakyReLU(alpha=alpha)(dense) # (t, N, N)
            
            #Mask values before activation
#            mask = -10e9 * (1.0 - A) # (N, N)
#            dense += mask # (t, N, N)
            
            #Apply softmax to get attention coefficients
#            dense = K.softmax(dense) # (t, N, N)
            
            #Apply dropout to features and attention coefficients
#            dropout_attn = Dropout(self.dropout_rate)(dense) # (t, N, N)
#            dropout_feat = Dropout(self.dropout_rate)(features) # (t, N, F')
            
            #Linear combination with neighbors' features
#            node_features = K.batch_dot(dropout_attn, dropout_feat) #(t, N, F')
            
#            if self.use_bias:
#                node_features = K.bias_add(node_features, self.biases[head]) # (t, N, F')
                
            #Add output of attention head to final output

#            outputs.append(node_features)
            
        #Aggregate the heads' output according to the reduction method
#        if self.attn_heads_reduction == 'concat':
#            output = K.concatenate(outputs) # (t, N, KF')
#        else:
#            output = K.mean(K.stack(outputs), axis=-1) # (t, N, F')
#        output = self.activation(output) # (t, N, KF') or (t, N, F')
#        return output
    
    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape
    

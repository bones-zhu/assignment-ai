#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:17:09 2020

@author: sh
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

class GraphAttentionLayer(keras.layers.Layer):
    def compute_output_signature(self, input_signature):
        pass
    
    def __init__(self, input_dim, output_dim, adj, nodes_num, dropout_rate=0.0, 
                 activation=None, use_bias=True, kernel_initalizer='glorot_uniform', 
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None, 
                 coef_dropout=0.0, **kwargs):
        """
        :param input_dim: 输入的维度
        :param output_dim: 输出的维度，不等于input_dim
        :param adj: 具有自环的tuple类型的邻接表[coords, values, shape], 可以采用sp.coo_matrix生成
        :param nodes_num: 节点数量
        :param dropout_rate: 丢弃率, 防过拟合，默认0.0
        :param activations: 激活函数
        :param use_bias: 偏移，默认True
        :param kernel_initalizer: 权值初始化方法
        :param bias_initializer: 偏置初始化方法
        :param kernel_regularizer: 权值正则化
        :param bias_regularizer: 输出正则化
        :param activity_regularizer: 输出正则化
        :param kernel_constraint: 权值约束
        :param bias_constraint: 偏移约束
        :param coef_dropout: 互相关系数丢弃，默认0.0
        """
        
        super(GraphAttentionLayer, self).__init__()
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initalizer = initializers.get(kernel_initalizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = initializers.get(kernel_regularizer)
        self.bias_regularizer = initializers.get(bias_regularizer)
        self.kernel_constraint = initializers.get(kernel_constraint)
        self.bias_constraint = initializers.get(bias_constraint)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.support = [tf.SparseTensor(indices=adj[0][0], values=adj[0][1], dense_shape=adj[0][2])]
        self.dropout_rate = dropout_rate
        self.coef_drop = coef_dropout
        self.nodes_num = nodes_num
        self.kernel = None
        self.mapping = None
        self.bias = None
        
    def build(self, input_shape):
        """
        只执行一次
        """
        self.kernel = self.add_weight(shape=(self.input_dim, self.output_dim), 
                                      initializers = self.kernel_initalizer, 
                                      regularizers = self.kernel_regularizer, 
                                      constraints = self.kernel_constraint, 
                                      trainable = True)
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.input_dim, 1), 
                                        initializers = self.bias_initalizer, 
                                        regularizers = self.bias_regularizer, 
                                        constraints = self.bias_constraint, 
                                        trainable = True)
        
        print('[GAT LAYER]: GAT W & b built.')
        
        
    def call(self, inputs, training=True):
        #完成输入到输出的映射关系
        # inputs = tf.nn.l2_normalize(inputs, 1)
        raw_shape = inputs.shape
        inputs = tf.reshape(inputs, shape=(1, raw_shape[0], raw_shape[1])) # (1, nodes_num, input_dim)
        mapped_inputs = keras.layers.Conv1D(self.output_dim, 1, use_bias=False)(inputs) #(1, nodes_num, output_dim)
        
        sa_1 = keras.layers.Conv1D(1, 1)(mapped_inputs) #(1, nodes_num, 1)
        sa_2 = keras.layers.Conv1D(1, 1)(mapped_inputs) # (1, nodes_num, 1)
        
        con_sa_1 = tf.reshape(sa_1, shape=(raw_shape[0], 1)) # (nodes_num, 1)
        con_sa_2 = tf.reshape(sa_2, shape=(raw_shape[0], 1)) # (nodes_num, 1)
        
        con_sa_1 = tf.cast(self.support[0], dtype=tf.float32) * con_sa_1 # (nodes_num, nodes_num) W_hi
        
        
        
        
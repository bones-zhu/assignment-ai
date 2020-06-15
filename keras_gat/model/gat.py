#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:11:13 2020

@author: sh
"""


import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop, bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat, out_sz=hid_units[0], activation=activation, in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
            
        
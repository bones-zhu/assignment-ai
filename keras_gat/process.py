#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:21:00 2020

@author: sh
"""


import numpy as np
import json
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
import pdb
import sys
sys.setrecursionlimit(99999)

def run_dfs(adj, msk, u, ind, nb_nodes):
    if msk[u] == -1:
        msk[u] = ind
        
        for v in adj[u, :].nonzero()[1]:
            run_dfs(adj, msk, v, ind, nb_nodes)
            
#Use depth-first search to split a graph into subgraphs
def dfs_split(adj):
    """Assume adj is of shape [nb_nodes, nb_nodes]"""
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.float32)
    
    graph_id = 0
    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1
            
    return ret

def test(adj, mapping):
    nb_nodes = adj.shape[0]
    for i in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] != mapping[j]:
                return False
    return True

def find_split(adj, mapping, ds_label):
    nb_nodes = adj.shape[0]
    dict_splits = {}
    for i in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] == 0 or mapping[j] == 0:
                dict_splits[0] = None
            elif mapping[i] == mapping[j]:
                if ds_label[i]['val'] == ds_label[j]['val'] and ds_label[i]['test'] == ds_label[j]['test']:
                    
                    if mapping[i] not in dict_splits.keys():
                        if ds_label[i]['val']:
                            dict_splits[mapping[i]] = 'val'
                        elif ds_label[i]['test']:
                            dict_splits[mapping[i]] = 'test'
                        else:
                            dict_splits[mapping[i]] = 'train'
                    else:
                        if ds_label[i]['test']:
                            ind_label = 'test'
                        elif ds_label[i]['val']:
                            ind_label = 'val'
                        else:
                            ind_label = 'train'
                        if dict_splits[mapping[i]] != ind_label:
                            print('inconsistent labels within a graph exiting!')
                            return None
    return dict_splits


    



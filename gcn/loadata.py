# -*- coding: utf-8 -*-
"""
@Time    : 10/30/19 9:51 AM
@Author  : Wang Zhiyuan
@Email   : wangzhiyuan@geotmt.com
@File    : generate_data.py
@Project: gcn
"""
import pandas as pd
import numpy as np
import random
import networkx as nx
from itertools import combinations
from sklearn.preprocessing import LabelBinarizer
import scipy.sparse as sp
import pickle as pkl
import sys
from utils import parse_index_file

def generate_data():
    sample_num = 2000
    features_num = 500
    mask = int(sample_num*0.3)*["train"]+int(sample_num*0.4)*["valid"]+int(sample_num-int(sample_num*0.3)-int(sample_num*0.4))*["test"]
    np.random.shuffle(mask)
    features = np.random.randint(0, 100, [sample_num, features_num])
    # 构建label
    cates = [1,2,3,4]
    labels = np.array([random.choice(cates) for j in range(0,sample_num)])
    data = pd.DataFrame(features,columns=['v'+str(i) for i in range(features_num)])
    data["gid"] = list(range(0+1000, sample_num+1000))
    data["type"] = mask
    data["y_label"] = labels
    data.to_csv("data/features.csv", index=None)
    return data

def create_graph():
    edges = 50000
    sample_num = 2000
    G = nx.Graph()
    # 构建边
    comb = list(combinations(list(range(0+1000, sample_num+1000)), 2))
    np.random.shuffle(comb)
    for c in comb[0:edges]:
        G.add_edge(*c, weight=random.uniform(0, 1))
    return G

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def mask_signal(signal,mask):
    mask_list = []
    for t in mask:
        if t == signal:
            mask_list.append(True)
        else:
            mask_list.append(False)
    return mask_list

def label_process(labels, mask):
    train_mask = mask_signal("train", mask)
    val_mask = mask_signal("valid", mask)
    test_mask = mask_signal("test", mask)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return train_mask, val_mask, test_mask, y_train, y_val, y_test




def prepare(data,G):
    adj = nx.adjacency_matrix(G, nodelist=list(data["gid"]))
    label_col = data["y_label"]
    mask = data["type"]
    data_v = data.drop(["gid", "y_label", "type"], axis=1)
    features = sp.lil_matrix(data_v.values)
    onehot = LabelBinarizer()
    labels = onehot.fit_transform(label_col)
    train_mask, val_mask, test_mask, y_train, y_val, y_test = label_process(labels, mask)
    return adj, features, train_mask, val_mask, test_mask, y_train, y_val, y_test

def gen_fea_graph(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    ng = {}
    for k in graph:
        v_l = []
        for v in graph[k]:
            v_l.append(v+1000)
        ng[k+1000] = v_l

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(ng)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    # 根据label的总行数，将生成训练集，验证集和测试集的过滤的index
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # 根据上一步生成的index，过滤生成训练集、验证集和测试集的标签
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    f_values = features.toarray()
    feadf = pd.DataFrame(f_values, columns=["v"+str(i) for i in range(f_values.shape[1])])
    feadf["gid"] = list(range(1000, f_values.shape[0]+1000))
    feadf["type"] = ["train"]*120 + ["valid"]*500 + ["test"]*(labels.shape[0]-620)
    label_list = []
    for l in labels:
        if 1 in l:
            label_list.append(list(l).index(1)+1)
        else:
            label_list.append(0)
    feadf["y_label"] = label_list
    feadf.to_csv("data/features.csv", index=None)
    pkl.dump(G, open("data/graph.pkl", "wb"))


# data = generate_data()
# gen_fea_graph('cora')
"""
读取数据,数据包含:"gid","y_label","type"
构建图谱必须以gid列作为节点名称
"""
data = pd.read_csv("data/features.csv")
# G = create_graph()
G = pkl.load(open("data/graph.pkl", "rb"))
adj, features, train_mask, val_mask, test_mask, y_train, y_val, y_test = prepare(data, G)

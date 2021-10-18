from __future__ import print_function
from posixpath import split
from types import LambdaType

import numpy as np
import matplotlib
from numpy.core.fromnumeric import transpose
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=5
N_WALKS=50

def dump_json(data, perfix, file_name):
  out_file = open(os.path.join(perfix, file_name + ".json"), 'w')
  json.dump(data, out_file)
  out_file.close()


def load_json(prefix, file_name):
  file_path = os.path.join(prefix, file_name + ".json")
  with open(file_path, "r") as read_file:
    print("loading", file_name)
    data_json = json.load(read_file)
  return data_json


def load_npy(prefix, file_name):
  file_path = os.path.join(prefix, file_name + ".npy")
  data_npy = np.load(file_path)
  return data_npy


def load_graph(prefix, graph_id):
  file_path = os.path.join(prefix, "graphs", str(graph_id) + ".json")
  G_data = json.load(open(file_path))
  G = json_graph.node_link_graph(G_data)
  return G

combined_ul = np.array([1600, 400, 480, 1600, 1, 1, 1, 25, 7, 3, 1, 1, 7, 7, 3, 1, 1, 1])
combined_node_feature_dict = {'1': 1, 'dilation_h_factor': 5, 'filter_height': 7, '0': 0, 'asymmetric_quantize_inputs': 15, 'padding': 10, '2': 2, 'dilation_w_factor': 6, 'keep_num_dims': 16, 'axis': 14, 'filter_width': 8, 'stride_h': 12, 'weights_format': 17, 'pot_scale_int16': 11, 'fused_activation_function': 9, '3': 3, 'depth_multiplier': 4, 'stride_w': 13}

def conditional_fit(prefix, feature_array, node_feature_dict, w, p):
  conditional_dict = {}
  conditional_dict["max"] = np.max(feature_array, axis=0)
  conditional_dict["min"] = np.min(feature_array, axis=0)
  conditional_dict["w"] = w
  conditional_dict["p"] = p
  ul = np.zeros(feature_array.shape[1])
  for op in node_feature_dict.keys():
    ul[node_feature_dict[op]] = combined_ul[combined_node_feature_dict[op]]
  conditional_dict["ul"] = ul
  pickle.dump(conditional_dict, open(os.path.join(prefix, "conditional.pickle"), "wb"))

  print("max:", conditional_dict["max"])
  print("min:", conditional_dict["min"])
  print("ul:", conditional_dict["ul"])

  return conditional_dict


def conditional_transform(feature_array, feature_mask, conditional_dict):
  feature_array_new = np.zeros((feature_array.shape[0], feature_array.shape[1]*2))
  max = conditional_dict["max"]
  min = conditional_dict["min"]
  ul = conditional_dict["ul"]
  w = conditional_dict["w"]
  p = conditional_dict["p"]
  for i in range(feature_mask.shape[0]):
    for j in range(feature_mask.shape[1]):
      if feature_mask[i, j] == 1:
        feature_array_new[i, 2*j] = w * np.sin(np.pi * p * feature_array[i, j] / ul[j])
        feature_array_new[i, 2*j+1] = w * np.cos(np.pi * p * feature_array[i, j] / ul[j])

  print("feature_array max:", np.max(feature_array_new, axis=0))
  print("feature_array min:", np.min(feature_array_new, axis=0))
  return feature_array_new


def load_data(prefix, normalize=True, conditional=False, w=2, p=1):
  dataset_name = prefix.split("/")[-2]
  print("Dataset:", dataset_name)

  data_json = load_json(prefix, "op_dict")
  op_dict = {int(k):str(v) for k,v in data_json.items()}
  print(op_dict)
  op_list = load_json(prefix, "op_list")
  feature_array = load_npy(prefix, "feature_array")
  data_json = load_json(prefix, "graph_dict")
  graph_dict = {str(k):v for k,v in data_json.items()}
  
  data_json = load_json(prefix, "node_dict")
  graph_node_max = 0
  node_dict = {}
  for k, v in data_json.items():
    node_dict[int(k)] = v 
    graph_node_max = max(graph_node_max, len(v))

  train_gids = graph_dict["train"]
  train_nids = []
  for gid in train_gids:
    train_nids += node_dict[gid]

  data_json = load_json(prefix, "node_feature_dict")
  node_feature_dict = {str(k):int(v) for k,v in data_json.items()}
  print(node_feature_dict)

  print("loading graphs")
  graph_count = len(node_dict)
  Gs = []
  for i in range(graph_count):
    Gs.append(load_graph(prefix, i))

  prediction_dict = {}
  data_json = load_json(prefix, "prediction_dict")
  for k, v in data_json.items():
    v = np.array(v) * 1000
    prediction_dict[str(k)] = v
  
  if normalize:
    train_feats = feature_array[train_nids]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    pickle.dump(scaler, open(os.path.join(prefix, "scaler.pickle"), "wb"))
    if conditional:
      conditional_dict = conditional_fit(prefix, feature_array, node_feature_dict, w, p)
      feature_mask = load_npy(prefix, "feature_mask_array")
      feature_array = conditional_transform(feature_array, feature_mask, conditional_dict)
    else:
      feature_array = scaler.transform(feature_array)

  op_dict[-1] = len(op_dict)
  print("#NN:", len(Gs))
  print("graph_node_max:", graph_node_max)
  return Gs, op_dict, op_list, feature_array, graph_dict, node_dict, graph_node_max, prediction_dict, node_feature_dict


def load_datas(prefixs, prefix_train, normalize=True, conditional=False, w=2, p=1):
  Gs = []
  op_dict = {}
  op_dict_reverse = {}
  op_list = np.zeros((0))
  feature_array = np.zeros((0,0))
  feature_mask = np.zeros((0,0))
  split_names = ["train", "val", "test"]
  graph_dict = {}
  for name in split_names:
    graph_dict[name] = np.zeros((0))
  node_dict = {}
  graph_node_max = 0
  prediction_names = ["inference_time"]
  prediction_dict = {}
  for name in prediction_names:
    prediction_dict[name] = np.zeros((0))
  node_feature_dict = {}
  n_count = 0
  g_count = 0
  op_count = 0
  feature_count = 0

  for prefix in prefixs:
    input_data = list(load_data(prefix, normalize=False))
    Gs = Gs+input_data[0]

    input_op_dict = input_data[1]
    op_convert = np.zeros(len(input_op_dict))
    for idx, op in input_op_dict.items():
      if idx != -1:
        if op not in op_dict_reverse.keys():
          op_dict_reverse[op] = op_count
          op_dict[op_count] = op
          op_count += 1
        op_convert[idx] = op_dict_reverse[op]
    input_op_list = input_data[2]
    print("input op_dict:", input_op_dict)
    print("op_dict:", op_dict)
    print("op_convert:", op_convert)
    input_op_list = op_convert[np.array(input_op_list)]
    print("converted op_list min:", min(input_op_list))
    print("converted op_list max:", max(input_op_list))
    op_list = np.hstack((op_list, input_op_list)).astype(int)

    input_graph_dict = input_data[4]
    for name in split_names:
      graph_dict[name] = np.hstack((graph_dict[name], np.array(input_graph_dict[name])+g_count)).astype(int)

    input_node_dict = input_data[5]
    for idx, nodes in input_node_dict.items():
      node_dict[idx+g_count] = np.array(nodes).astype(int) + n_count

    graph_node_max = max(graph_node_max, input_data[6])
    for name in prediction_names:
      prediction_dict[name] = np.hstack((prediction_dict[name], input_data[7][name]))

    input_node_feature_dict = input_data[8]
    for f in sorted(input_node_feature_dict.keys()):
      if f not in node_feature_dict.keys():
        print("Feature Name:", f, "Feature Index:", feature_count)
        node_feature_dict[f] = feature_count
        feature_count += 1
    feature_array_old = np.zeros((feature_array.shape[0], feature_count))
    feature_array_old[:,0:feature_array.shape[1]] = feature_array
    feature_mask_old = np.zeros((feature_mask.shape[0], feature_count))
    feature_mask_old[:,0:feature_mask.shape[1]] = feature_mask

    input_feature_array = input_data[3]
    feature_array_new = np.zeros((input_feature_array.shape[0], feature_count))
    input_feature_mask = load_npy(prefix, "feature_mask_array")
    feature_mask_new = np.zeros((input_feature_array.shape[0], feature_count))
    for f in input_node_feature_dict:
      input_index = input_node_feature_dict[f]
      index = node_feature_dict[f]
      feature_array_new[:, index] = input_feature_array[:, input_index]
      feature_mask_new[:, index] = input_feature_mask[:, input_index]
    feature_array = np.vstack((feature_array_old, feature_array_new))
    feature_mask = np.vstack((feature_mask_old, feature_mask_new))
      
    n_count += len(input_data[2])
    g_count += len(input_data[5])

  if not os.path.exists(prefix_train):
    os.makedirs(prefix_train)
  dump_json(op_dict, prefix_train, "op_dict")
  dump_json(node_feature_dict, prefix_train, "node_feature_dict")

  #'''
  print("G length:", len(Gs))
  print("op_dict:", op_dict)
  print("op_list length:", len(op_list))
  print("feature_array shape:", feature_array.shape)
  print("feature_array max:", np.max(feature_array, axis=0))
  print("feature_array min:", np.min(feature_array, axis=0))
  for name in split_names:
    print("graph_dict "+name+" length:", len(graph_dict[name]))
    print(name+" max:", max(graph_dict[name]))
    print(name+" min:", min(graph_dict[name]))
  print("node_dict length:", len(node_dict))
  print("graph_node_max:", graph_node_max)
  for name in prediction_names:
    print("prediction_dict "+name+" length:", len(prediction_dict[name]))
  print("node_feature_dict:", node_feature_dict)
  #'''

  if normalize:
    train_nids = graph_dict["train"]
    train_feats = feature_array[train_nids]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    pickle.dump(scaler, open(os.path.join(prefix_train, "scaler.pickle"), "wb"))
    if conditional:
      conditional_dict = conditional_fit(prefix_train, feature_array, node_feature_dict, w, p)
      #feature_mask = load_npy(prefix_train, "feature_mask_array")
      feature_array = conditional_transform(feature_array, feature_mask, conditional_dict)
    else:
      feature_array = scaler.transform(feature_array)
  op_dict[-1] = len(op_dict)

  return Gs, op_dict, op_list, feature_array, graph_dict, node_dict, graph_node_max, prediction_dict, node_feature_dict


def load_data_extrapolate(prefix, prefix_train, normalize=True, conditional=False):
  # Load train dicts
  data_json = load_json(prefix_train, "op_dict")
  op_dict_reverse_train = {str(v):int(k) for k,v in data_json.items()}
  data_json = load_json(prefix_train, "node_feature_dict")
  node_feature_dict_train = {str(k):v for k,v in data_json.items()}
  #feature_array_train = load_npy(prefix_train, "feature_array")
  print(op_dict_reverse_train)
  print(node_feature_dict_train)

  # Load input data
  input_data = list(load_data(prefix, normalize=False))
  op_dict = input_data[1]
  op_list = np.array(input_data[2])
  feature_array = input_data[3]
  node_feature_dict = input_data[8]

  op_convert = np.zeros(len(op_dict))
  for key, op in op_dict.items():
    if key != -1:
      index_train = op_dict_reverse_train[op]
      op_convert[key] = index_train
  input_data[2] = op_convert[op_list]

  op_dict[-1] = len(op_dict_reverse_train)
  input_data[1] = op_dict

  feature_array_new = np.zeros((feature_array.shape[0], len(node_feature_dict_train)))
  feature_mask_new = np.zeros((feature_array.shape[0], len(node_feature_dict_train)))
  feature_mask = load_npy(prefix, "feature_mask_array")
  for f in node_feature_dict:
    index = node_feature_dict[f]
    index_train = node_feature_dict_train[f]
    feature_array_new[:, index_train] = feature_array[:, index]
    feature_mask_new[:, index_train] = feature_mask[:, index]
  print("feature_array_new max:", np.max(feature_array_new, axis=0))
  print("feature_array_new min:", np.min(feature_array_new, axis=0))
  if normalize:
    scaler = pickle.load(open(os.path.join(prefix_train, "scaler.pickle"),"rb"))
    if conditional:
      conditional_dict = pickle.load(open(os.path.join(prefix_train, "conditional.pickle"),"rb"))
      feature_array_new = conditional_transform(feature_array_new, feature_mask_new, conditional_dict)
    else:
      feature_array_new = scaler.transform(feature_array_new)

  input_data[3] = feature_array_new

  return input_data


def main(argv):
  prefix = "../data/cifar10_july8_mask/"
  #load_data(prefix, normalize=True, conditional=True)
  prefixs = ["../data/cifar10_july8_mask/", "../data/imagenet_may26/", "../data/kws_chu_july19/", "../data/super/", "../data/emnist_28/", "../data/emnist_64/", "../data/ssd_80_v3/"]
  prefixs = ["../data/cifar10_july8_v3/", "../data/imagenet_may26_v3/", "../data/kws_chu_july19_v3/", "../data/super_v3/", "../data/emnist_28_v3/", "../data/emnist_64_v3/"]#, "../data/ssd_80_v3/"]
  prefixs = ["../data/kws_chu_july19_v3/", "../data/super_v3/", "../data/emnist_28_v3/", "../data/emnist_64_v3/"]
  prefix_train = "../data/EC_EI_KWS_SR_EM28_EM64_v3/"
  load_datas(prefixs, prefix_train, normalize=True, conditional=False, w=2, p=1)

if __name__ == "__main__":
  main(sys.argv)

from pickle import NONE
from packaging import version

import json
import pickle
import csv
import os
import re
import sys
import numpy as np
import random
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt

from tensorflow.lite.python import schema_py_generated as schema_fb


def BuiltinCodeToName(code):
  """Converts a builtin op code enum to a readable name."""
  for name, value in schema_fb.BuiltinOperator.__dict__.items():
    if value == code:
      return name
  return None


def NameListToString(name_list):
  """Converts a list of integers to the equivalent ASCII string."""
  if isinstance(name_list, str):
    return name_list
  else:
    result = ""
    if name_list is not None:
      for val in name_list:
        result = result + chr(int(val))
    return result


def CamelCaseToSnakeCase(camel_case_input):
  """Converts an identifier in CamelCase to snake_case."""
  s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_input)
  return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def CreateDictFromFlatbuffer(buffer_data):
  model_obj = schema_fb.Model.GetRootAsModel(buffer_data, 0)
  model = schema_fb.ModelT.InitFromObj(model_obj)
  return FlatbufferToDict(model, preserve_as_numpy=False)


def FlatbufferToDict(fb, preserve_as_numpy):
  """Converts a hierarchy of FB objects into a nested dict.
  We avoid transforming big parts of the flat buffer into python arrays. This
  speeds conversion from ten minutes to a few seconds on big graphs.
  Args:
    fb: a flat buffer structure. (i.e. ModelT)
    preserve_as_numpy: true if all downstream np.arrays should be preserved.
      false if all downstream np.array should become python arrays
  Returns:
    A dictionary representing the flatbuffer rather than a flatbuffer object.
  """
  if isinstance(fb, int) or isinstance(fb, float) or isinstance(fb, str):
    return fb
  elif hasattr(fb, "__dict__"):
    result = {}
    for attribute_name in dir(fb):
      attribute = fb.__getattribute__(attribute_name)
      if not callable(attribute) and attribute_name[0] != "_":
        snake_name = CamelCaseToSnakeCase(attribute_name)
        preserve = True if attribute_name == "buffers" else preserve_as_numpy
        result[snake_name] = FlatbufferToDict(attribute, preserve)
    return result
  elif isinstance(fb, np.ndarray):
    return fb if preserve_as_numpy else fb.tolist()
  elif hasattr(fb, "__len__"):
    return [FlatbufferToDict(entry, preserve_as_numpy) for entry in fb]
  else:
    return fb


class OpCodeMapper(object):
  """Maps an opcode index to an op name."""

  def __init__(self, data):
    self.code_to_name = {}
    for idx, d in enumerate(data["operator_codes"]):
      self.code_to_name[idx] = BuiltinCodeToName(d["deprecated_builtin_code"])
      if self.code_to_name[idx] == "CUSTOM":
        self.code_to_name[idx] = NameListToString(d["custom_code"])

  def __call__(self, x):
    if x not in self.code_to_name:
      s = "<UNKNOWN>"
    else:
      s = self.code_to_name[x]
    return "%s (%d)" % (s, x)

  def get_opcodes(self):
    return self.code_to_name.keys()

  def get_ops(self):
    return self.code_to_name.values()

  def get_op(self, opcode):
    return self.code_to_name[opcode]


class ComputeGraphs(object):
  """Create a collection of compute graphs base on saved tflite model."""

  def __init__(self, io_flag, input_dir, output_dir, v3_flag):
    self.io_flag = io_flag
    self.input_dir = input_dir
    self.output_dir = output_dir
    if v3_flag:
      self.output_dir = output_dir + "_v3"
    self.output_graphs_dir = os.path.join(self.output_dir, "graphs")
    if not os.path.exists(self.output_graphs_dir):
      os.makedirs(self.output_graphs_dir)

    # Init all parameters
    self.op_dict = {0:"input", 1:"output"}
    self.op_dict_r = {}
    self.op_list = []
    self.feature_vec_list = []
    self.feature_mask_list = []
    self.node_feature_dict = {}
    self.node_dict = {}
    self.graph_count = 0
    self.node_count = 0
    self.node_count_list = []
    self.G = None
    self.prediction_dict = {}
    self.edge_feature_length = 4
    self.node_feature_length_max = 20
    self.val_percent = 0.1
    self.test_percent = 0.1

    # Init node_feature_dict with tensor output size
    for i in range(self.edge_feature_length):
      self.node_feature_dict[i] = i

  def _get_edge_features(self, edge):
    shape = edge["shape"]
    for i in range(self.edge_feature_length - len(shape)):
      shape.insert(0, 1)
    return shape

  def _get_feature_vec(self, features, edge_features):
    vec = np.zeros(self.node_feature_length_max)
    mask = np.zeros(self.node_feature_length_max)
    #print(features)
    if features is not None:
      for key, value in features.items():
        if key not in self.node_feature_dict.keys():
          self.node_feature_dict[key] = len(self.node_feature_dict)
        idx = self.node_feature_dict[key]
        vec[idx] = int(value)
        mask[idx] = 1

    for d in range(len(edge_features)):
      idx = self.node_feature_dict[d]
      vec[idx] = edge_features[d]
      mask[idx] = 1
      
    return vec, mask

  def _add_subgraph_node(self, opcode, features, edge_features):
    op_index = self.node_count_list[-1]
    self.G.add_node(op_index, opcode=opcode, features=features)
    self.op_list.append(self.op_dict_local[opcode])
    feature_vec, feature_mask = self._get_feature_vec(features, edge_features)
    self.feature_vec_list.append(feature_vec)
    self.feature_mask_list.append(feature_mask)
    self.node_dict[self.graph_count].append(self.node_count)
    self.node_count += 1
    self.node_count_list[-1] += 1
    return op_index

  def _add_subgraph(self, g):
    edges = {}
    shape_length = 4
    
    # Add in all the nodes (operators) into the graph, also add in all the edges' connections
    for op_idx, op in enumerate(g["operators"] or []):
      edge_features = []
      #print(op_index, op)
      output_tensor = g["tensors"][op["outputs"][0]]
      edge_features = self._get_edge_features(output_tensor)
      op_index = self._add_subgraph_node(op["opcode_index"], op["builtin_options"], edge_features)

      for tensor_input_position, tensor_index in enumerate(op["inputs"]):
        if tensor_index not in edges:
          edges[tensor_index] = [[], []]
        edges[tensor_index][1].append(op_index)

      for tensor_output_position, tensor_index in enumerate(op["outputs"]):
        if tensor_index not in edges:
          edges[tensor_index] = [[], []]
        edges[tensor_index][0].append(op_index)
        
    # Add in all the edges (tensors) into the graph
    for tensor_index, tensor in enumerate(g["tensors"]):
      #print(tensor_index, tensor["shape"])
      shape = self._get_edge_features(tensor)
      features_dict = {'shape': shape}
      sources = edges[tensor_index][0]
      targets = edges[tensor_index][1]
      
      # Add in input node and connections
      if len(sources) == 0:
        op_index = self._add_subgraph_node(-1, {}, shape)
        sources.append(op_index)
      #print("sources", sources)
      #print("targets", targets)

      '''
      # Add in output node and connections
      if tensor_index == len(g["tensors"]) - 1:
        self._add_subgraph_node(-2, -2, {}, shape)
      if len(targets) == 0:
        targets.append(-2)
      '''

      for source in sources:
        for target in targets:
          if (source >= 0 and target >= 0) or self.io_flag:
            self.G.add_edge(source, target, features=features_dict)
            #print(tensor_index, source, target, shape)

  def _load_prediction(self, f_dict_path):
    with open(f_dict_path, newline='') as f:
      reader = csv.reader(f)
      f_data = list(reader)
    keys = f_data[0]
    values = f_data[1]
    f_dict = {keys[index]:values[index] for index in range(len(keys))}
    #print(f_dict)
    if "inference_time" not in self.prediction_dict.keys():
      self.prediction_dict['inference_time'] = []
    self.prediction_dict['inference_time'].append(float(f_dict['inference_time']))

  def _create_graph_dict(self, graph_count, val_percent, test_percent):
    graph_ids = range(graph_count)
    val_count = int(val_percent*graph_count)
    test_count = int(test_percent*graph_count)
    val_ids = random.sample(graph_ids, val_count+test_count)
    train_ids = [x for x in graph_ids if x not in val_ids]
    test_ids = random.sample(val_ids,  test_count)
    val_ids = [x for x in val_ids if x not in test_ids]

    train_ids.sort()
    val_ids.sort()
    test_ids.sort()

    return {"train": train_ids, "val": val_ids, "test": test_ids}

  def _dump_json(self, data, file_name):
    out_file = open(os.path.join(self.output_dir, file_name + ".json"), 'w')
    json.dump(data, out_file)
    out_file.close()

  def _dump_pickle(self, data, file_name):
    out_file = open(os.path.join(self.output_dir, file_name + ".pickle"), 'wb')
    pickle.dump(data, out_file)
    out_file.close()

  def _dump_npy(self, data, file_name):
    file_dir = os.path.join(self.output_dir, file_name)
    np.save(file_dir, data)

  def add_graph_tflite(self, tflite_name, v3_flag):
    # Check if the model and its csv file exits
    self.tflite_dir = os.path.join(self.input_dir, tflite_name)
    model_path = os.path.join(self.tflite_dir, "tflites", tflite_name+"_INT8.tflite")
    f_dict_path = os.path.join(self.tflite_dir, "tflites", "output", tflite_name + "_INT8_summary_internal-default.csv")
    if v3_flag:
      f_dict_path = os.path.join(self.tflite_dir, "tflites", "output_v3", tflite_name + "_INT8_summary_internal-default.csv")

    #'''
    if not (os.path.exists(model_path) and os.path.exists(f_dict_path)):
      #print("This model is broken:", tflite_name)
      #if not os.path.exists(model_path):
      #  print("Model Path:", model_path, "is missing.")
      #if not os.path.exists(f_dict_path):
      #  print("Vela Path:", f_dict_path, "is missing.")
      return
    #'''

    # Extract the data from tflite_input
    if model_path.endswith(".tflite"):
      with open(model_path, "rb") as file_handle:
        file_data = bytearray(file_handle.read())
      try:
        data = CreateDictFromFlatbuffer(file_data)
      except:
        print("Data is broken:", tflite_name)
        return
    elif model_path.endswith(".json"):
      data = json.load(open(model_path))
    else:
      raise RuntimeError("Input file was not .tflite or .json")

    # Init variables for this model
    self.node_count_list.append(0)
    self.node_dict[self.graph_count] = []

    # Add in all the ops in this data to op_dict and local op_dict
    opcode_mapper = OpCodeMapper(data)
    ops = list(opcode_mapper.get_ops())
    opcodes = list(opcode_mapper.get_opcodes())
    self.op_dict_local = {-1:0, -2:1}
    
    for opcode in opcodes:
      op = opcode_mapper.get_op(opcode)
      if op not in self.op_dict.values():
        print("Added Operator:", op)
        key = len(self.op_dict)
        self.op_dict[key] = op
        self.op_dict_r[op] = key
      self.op_dict_local[opcode] = self.op_dict_r[op]
    
    #print(self.op_dict_local)
    
    # Create the graph base in NetworkX format.
    self.G = nx.DiGraph()
    for subgraph_idx, g in enumerate(data["subgraphs"]):
      self._add_subgraph(g)

    # Write graph data to json
    G_data = json_graph.node_link_data(self.G)
    file_name = "graphs/" + str(self.graph_count)
    self._dump_json(G_data, file_name)

    # Load predictions
    self._load_prediction(f_dict_path)

    # Increment the counter
    self.graph_count += 1

  def save(self):
    #self.op_dict["op_lict"] = self.op_list
    self._dump_json(self.op_dict, "op_dict")
    self._dump_json(self.op_list, "op_list")

    #self.node_dict["graph_count"] = self.graph_count
    #self.node_dict["node_count"] = self.node_count
    #self.node_dict["node_count_list"] = self.node_count_list
    self.graph_dict = self._create_graph_dict(self.graph_count, self.val_percent, self.test_percent)
    self._dump_json(self.graph_dict, "graph_dict")
    self._dump_json(self.node_dict, "node_dict")

    self.feature_vec_array = np.vstack(self.feature_vec_list)
    self.feature_vec_array = self.feature_vec_array[:, 0:len(self.node_feature_dict)]
    self._dump_npy(self.feature_vec_array, "feature_array")
    self.feature_mask_array = np.vstack(self.feature_mask_list)
    self.feature_mask_array = self.feature_mask_array[:, 0:len(self.node_feature_dict)]
    self._dump_npy(self.feature_mask_array, "feature_mask_array")
    self._dump_json(self.node_feature_dict, "node_feature_dict")
    #print("feature_vec_array shape:", self.feature_vec_array.shape)
    #print("feature_vec_array:", feature_vec_array)

    self._dump_json(self.prediction_dict, "prediction_dict")

  def print(self):
    print("op_dict:", self.op_dict)
    #print("op_list:", self.op_list)
    print("node_feature_dict:", self.node_feature_dict)
    #print("feature_vec_list:", self.feature_vec_list)
    #print("node_dict:", self.node_dict)
    print("graph_count:", self.graph_count)
    print("node_count:", self.node_count)
    #print("node_count_list:", self.node_count_list)
    #print("prediction_dict:", self.prediction_dict)

  def plot(self, output_dir):
    nx.draw(self.G)  # networkx draw()
    plt.draw()  # pyplot draw() 
    plt.savefig(output_dir, dpi=300, bbox_inches='tight')
    plt.show()


def main(argv):
  io_flag = True
  input_names = ["efficientnet_cifar10_quantization", "efficientnet_cifar10_quantization", "imagenet_may26", "cifar10_july8", "kws_chu_july19", "cifar10_july8", "super", "emnist_28", "emnist_64", "ssd_80"]
  output_names = ["efficientnet_cifar10_quantization", "efficientnet_cifar10_quantization_1k", "imagenet_may26", "cifar10_july8", "kws_chu_july19", "cifar10_july8_mask", "super", "emnist_28", "emnist_64", "ssd_80_v3"]
  index = 9
  v3_flag = False
  input_name = input_names[index]
  output_name = output_names[index]
  print("Input:", input_name, "\nOutput:", output_name)

  input_dir = "../../datasets/" + input_name
  tflite_name_test = "efficientnet_cifar10_seed1000_tf_use_stats"
  output_dir = "../data/" + output_name
  tflite_names = next(os.walk(input_dir))[1]

  Gs = ComputeGraphs(io_flag, input_dir, output_dir, v3_flag)
  for i, tflite_name in enumerate(tflite_names):
    Gs.add_graph_tflite(tflite_name, v3_flag)
    if (i + 1) % 1000 == 0:
      print(i + 1, "inputs converted!")
      #break
    #break
  Gs.print()
  Gs.save()
  

if __name__ == "__main__":
  main(sys.argv)
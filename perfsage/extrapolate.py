from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from minibatch import GraphMinibatchIterator
from utils import load_data, load_data_extrapolate
from supervised_train import log_dir, incremental_evaluate, get_num_params

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('extrapolate_prefix', '', 'prefix identifying training data. must be specified.')
flags.DEFINE_string('load_model_step', '', 'prefix identifying training data. must be specified.')

def plot_dots(x, y, xname, yname, title):
  n_bins = 100
  f_size = 16
  dots = np.linspace(0, 10000, 10000)
  zeros = np.zeros(10000)

  fig, axs = plt.subplots(3, 1, figsize=(6,12), gridspec_kw={'height_ratios': [2, 1, 1]})
  axs[0].plot(x, y, '.')
  axs[0].plot(dots, zeros, '--', color='r')
  axs[0].set_xlabel(xname, fontsize=f_size)
  axs[0].set_ylabel(yname, fontsize=f_size)
  axs[0].set_xlim([0,50])
  axs[0].set_ylim([-100,100])
  axs[0].set_title(title, fontsize=f_size)
  axs[1].hist(x, bins=n_bins)
  axs[1].set_xlim([0,50])
  axs[1].set_ylim([0,3000])
  axs[1].set_ylabel("Count", fontsize=f_size)
  axs[2].hist(y, bins=n_bins)
  axs[2].plot(zeros, dots, '--', color='r')
  axs[2].set_xlim([-50,50])
  axs[2].set_ylim([0,6000])
  axs[2].set_ylabel("Count", fontsize=f_size)
  axs[2].set_xlabel(yname, fontsize=f_size)

  save_dir = os.path.join(log_dir(), "plots")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  save_path = os.path.join(save_dir, title + "_" + xname + "_VS_" + yname + ".png")
  fig.savefig(save_path, bbox_inches='tight')
  plt.close()

def plot_box(x, xname, yname, title):
  fig, ax = plt.subplots()
  ax.boxplot(x)
  ax.set_xlabel(xname)
  ax.set_ylabel(yname)
  ax.set_ylim([-5,15])
  ax.set_title(title)
  save_dir = os.path.join(log_dir(), "plots")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  save_path = os.path.join(save_dir, title + "_box.png")
  fig.savefig(save_path, bbox_inches='tight')
  plt.close()

def extrapolate(extrapolate_data, load_path, load_step, train_name, extrapolate_name, show_summary=False):
  Gs = extrapolate_data[0]
  op_dict = extrapolate_data[1]
  op_list = extrapolate_data[2]
  features = extrapolate_data[3]
  graph_dict = extrapolate_data[4]
  node_dict = extrapolate_data[5]
  graph_node_max = FLAGS.max_graph_node_count
  if graph_node_max < extrapolate_data[6]:
        raise Exception("Maximum number of nodes in a graph is larger than FLAGS.max_graph_node_count")
  prediction_dict = extrapolate_data[7]

  node_count = len(op_list)
  num_preds = len(prediction_dict)
  op_count = op_dict[-1]

  if not features is None:
    # pad with dummy zero vector
    features = np.vstack([features, np.zeros((features.shape[1],))])
  ops = np.array(op_list)
  ops = np.hstack((ops, np.array([op_count]))).astype(int)

  sess=tf.Session()  
  sess.run(tf.global_variables_initializer())
  new_saver = tf.train.import_meta_graph(os.path.join(load_path, "model-"+str(load_step)+".meta"))
  new_saver.restore(sess, tf.train.latest_checkpoint(load_path))
  placeholders = {placeholder.name[0:-2]:placeholder for op in tf.get_default_graph().get_operations() if op.type=='Placeholder' for placeholder in op.values()}
  if show_summary:
    get_num_params()

  #placeholders = construct_placeholders(num_preds, features.shape[1], FLAGS.max_degree)
  minibatch = GraphMinibatchIterator(Gs, graph_dict, node_dict, node_count,
    placeholders, features, ops, prediction_dict, num_preds, graph_node_max,
    batch_size=FLAGS.batch_size, max_degree=FLAGS.max_degree)

  val_cost, val_error, val_errors, duration = incremental_evaluate(sess, minibatch, FLAGS.batch_size, load=True)
  #print("val_errors shape:", val_errors.shape)
  #print("graph_node_count shape:", graph_node_count.shape)
  #print("prediction_latency shape:", prediction_latency.shape)

  # Plotting 
  plot_title = train_name + "->" + extrapolate_name
  latency_error = val_errors[:,0]
  graph_node_count = np.array([len(v) for k,v in node_dict.items()])
  plot_dots(graph_node_count, latency_error*100, "graph_node_count", "latency MAPE(%)", plot_title)
  prediction_latency = np.array(prediction_dict["inference_time"])
  plot_dots(prediction_latency, latency_error*100, "latency", "latency MAPE(%)", plot_title)
  plot_box(features[:,0:-1], "Features", "Value", plot_title)

  print("Loaded Session", 
        "val_loss=", "{:.5f}".format(val_cost),
        "val_error=", "{:.5f}".format(val_error),
        "time=", "{:.5f}".format(duration))

  return 0

def main(argv=None):
  train_prefix = FLAGS.train_prefix
  train_name = train_prefix.split("/")[-2]
  print("Train Dataset:", train_name)
  extrapolate_prefix = FLAGS.extrapolate_prefix
  extrapolate_name = extrapolate_prefix.split("/")[-2]
  print("Extrapolate Dataset:", extrapolate_name)
  load_path = os.path.join(log_dir(), "models")
  load_step = FLAGS.load_model_step
  get_num_params = False

  print("Loading extrapolate data..")
  extrapolate_data = load_data_extrapolate(extrapolate_prefix, train_prefix, conditional=FLAGS.train_conditional)
  
  print("Extrapolating data..")
  extrapolate(extrapolate_data, load_path, load_step, train_name, extrapolate_name, get_num_params)

if __name__ == '__main__':
    tf.app.run()
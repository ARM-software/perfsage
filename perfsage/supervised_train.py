from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics

from supervised_models import SupervisedPerfSAGE
from models import SAGEInfo
from minibatch import GraphMinibatchIterator
from neigh_samplers import UniformNeighborSampler
from utils import load_data, load_datas

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')  
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')
flags.DEFINE_boolean('train_combined', False, "Whether to be combined dataset.")
flags.DEFINE_boolean('train_conditional', False, "Whether to use conditional normalization.")

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 200, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 50, 'maximum node degree.')
flags.DEFINE_integer('max_graph_node_count', 200, 'maximum number of nodes in a graph')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 64, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_integer('op_embedding_dim', 32, 'Set to positive value to use embedding dimension of op.')

#logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', './saves', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('save_threshold', 0.03, "The threshold to save the model.")
flags.DEFINE_integer('save_epoch', 50, "The epoch to save the model.")
flags.DEFINE_integer('validate_iter', -1, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 64, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 100, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 1

def get_num_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print(variable.name, shape, variable_parameters)
        total_parameters += variable_parameters
    print("Total Parameter Count:", total_parameters)

def predict_error(y_true, y_pred):
    errors = (y_true - y_pred) / y_true
    mape = np.average(np.abs(errors))
    error = mape
    return error, errors

# Define model evaluation function
def evaluate(sess, minibatch_iter, size=-1):
    t_test = time.time()
    feed_dict_val, preds = minibatch_iter.node_val_feed_dict(size)

    graph = tf.get_default_graph()
    model_preds = graph.get_tensor_by_name("predicts:0")
    mdoel_loss = graph.get_tensor_by_name("loss:0")
    outs_val = sess.run([model_preds, mdoel_loss], feed_dict=feed_dict_val)
    error, errors = predict_error(preds, outs_val[0])
    return outs_val[1], error, errors, (time.time() - t_test), errors

def incremental_evaluate(sess, minibatch_iter, size, test=False, load=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    preds = []
    iter_num = 0

    graph = tf.get_default_graph()
    model_preds = graph.get_tensor_by_name("predicts:0")
    mdoel_loss = graph.get_tensor_by_name("loss:0")
    while not finished:
        feed_dict_val, batch_preds, finished  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test, load=load)
        node_outs_val = sess.run([model_preds, mdoel_loss], feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        preds.append(batch_preds)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    preds = np.vstack(preds)
    error, errors = predict_error(preds, val_preds)
    return np.mean(val_losses), error, errors, (time.time() - t_test)

def log_dir():
    l2_power = 99
    if FLAGS.weight_decay != 0:
        l2_power = -np.log10(FLAGS.weight_decay)

    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}_{l2:02.0f}_{dropout:0.2f}".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate,
            l2=l2_power,
            dropout=FLAGS.dropout, 
            )
    if FLAGS.train_conditional:
        log_dir += "_c/"
    else:
        log_dir += "/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def get_model_name():
    models_dir = log_dir() + "models/"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_name = models_dir + "model"
    return model_name

def construct_placeholders(num_preds, num_features, max_degree):
    # Define placeholders
    placeholders = {
        'preds' : tf.placeholder(tf.float32, shape=(None, num_preds), name='preds'),
        'batch' : tf.placeholder(tf.int32, shape=(None), name='batch'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
        'features' : tf.placeholder(tf.float32, shape=(None, num_features), name='features'),
        'category_lookup' : tf.placeholder(tf.int32, shape=(None), name='category_lookup'),
        'adj' : tf.placeholder(tf.int32, shape=(None, max_degree), name='adj'),
    }
    return placeholders

def train(train_data, test_data=None):

    Gs = train_data[0]
    op_dict = train_data[1]
    op_list = train_data[2]
    features = train_data[3]
    graph_dict = train_data[4]
    node_dict = train_data[5]
    graph_node_max = FLAGS.max_graph_node_count
    if graph_node_max < train_data[6]:
        raise Exception("FLAGS.max_graph_node_count needs to be larger than", train_data[6])
    prediction_dict = train_data[7]

    node_count = len(op_list)
    num_preds = len(prediction_dict)
    op_count = op_dict[-1]

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])
    ops = np.array(op_list)
    ops = np.hstack((ops, np.array([op_count]))).astype(int)

    placeholders = construct_placeholders(num_preds, features.shape[1], FLAGS.max_degree)
    minibatch = GraphMinibatchIterator(Gs, graph_dict, node_dict, node_count,
            placeholders, features, ops, prediction_dict, num_preds, graph_node_max,
            batch_size=FLAGS.batch_size, max_degree=FLAGS.max_degree)
    #adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    #adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler()
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

        model = SupervisedPerfSAGE(num_preds, placeholders, 
                                     op_count,
                                     FLAGS.op_embedding_dim,
                                     minibatch.deg,
                                     layer_infos, 
                                     graph_node_max=graph_node_max,
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    elif FLAGS.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler()
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]

        model = SupervisedPerfSAGE(num_preds, placeholders, 
                                     op_count,
                                     FLAGS.op_embedding_dim,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     graph_node_max=graph_node_max,
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     concat=False,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler()
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedPerfSAGE(num_preds, placeholders, 
                                     op_count,
                                     FLAGS.op_embedding_dim,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     graph_node_max=graph_node_max,
                                     aggregator_type="seq",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler()
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedPerfSAGE(num_preds, placeholders, 
                                    op_count,
                                    FLAGS.op_embedding_dim,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     graph_node_max=graph_node_max,
                                     aggregator_type="maxpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler()
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedPerfSAGE(num_preds, placeholders, 
                                    op_count,
                                    FLAGS.op_embedding_dim,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     graph_node_max=graph_node_max,
                                     aggregator_type="meanpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
    saver = tf.train.Saver()
    model_name = get_model_name()
     
    # Init variables
    sess.run(tf.global_variables_initializer())
    
    # Train model
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    #train_adj_info = tf.assign(adj_info, minibatch.adj)
    #val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    for epoch in range(FLAGS.epochs): 
        minibatch.shuffle() 

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, preds = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
            train_cost = outs[2]

            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)
                train_error, train_errors = predict_error(preds, outs[-1])
                print("Epoch:", '%04d' % (epoch + 1), 
                    "Step:", '%04d' % total_steps, 
                    "train_loss=", "{:.5f}".format(train_cost),
                    "train_error=", "{:.5f}".format(train_error),  
                    "time=", "{:.5f}".format(avg_time))

            if FLAGS.validate_iter != -1 and total_steps % FLAGS.validate_iter == 0:
                val_cost, val_error, val_errors, duration = incremental_evaluate(sess, minibatch, FLAGS.validate_batch_size)
                #sess.run(train_adj_info.op)
                #epoch_val_costs[-1] += val_cost
                print("Epoch:", '%04d' % (epoch + 1), 
                    "Step:", '%04d' % total_steps, 
                    "val_loss=", "{:.5f}".format(val_cost),
                    "val_error=", "{:.5f}".format(val_error),
                    "time=", "{:.5f}".format(duration))

            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        # Validation
        #sess.run(val_adj_info.op)
        #val_cost, val_error, duration = evaluate(sess, model, minibatch, FLAGS.validate_batch_size)
        val_cost, val_error, val_errors, duration = incremental_evaluate(sess, minibatch, FLAGS.validate_batch_size)
        #sess.run(train_adj_info.op)
        epoch_val_costs[-1] += val_cost
        print("Epoch:", '%04d' % (epoch + 1), 
              "val_loss=", "{:.5f}".format(val_cost),
              "val_error=", "{:.5f}".format(val_error),
              "time=", "{:.5f}".format(duration))
        if val_error <= FLAGS.save_threshold or epoch % FLAGS.save_epoch == 0:
            print("Saving the model @", total_steps)
            saver.save(sess, model_name, global_step=total_steps)

        if total_steps > FLAGS.max_total_steps:
            break
    
    print("Optimization Finished!")
    print("Saving the model @", total_steps)
    saver.save(sess, model_name, global_step=total_steps)

    val_cost, val_error, val_errors, duration = incremental_evaluate(sess, minibatch, FLAGS.batch_size)
    print("Validation:", 
        "val_loss=", "{:.5f}".format(val_cost),
        "val_error=", "{:.5f}".format(val_error),
        "time=", "{:.5f}".format(duration))

    val_cost, val_error, val_errors, duration = incremental_evaluate(sess, minibatch, FLAGS.batch_size, test=True)
    print("Test:", 
        "test_loss=", "{:.5f}".format(val_cost),
        "test_error=", "{:.5f}".format(val_error),
        "time=", "{:.5f}".format(duration))

def main(argv=None):
    print("Loading training data..")
    if FLAGS.train_combined:
        train_prefixs = ["./data/cifar10_july8_mask/", "./data/imagenet_may26/", "./data/kws_chu_july19/", "./data/super/", "./data/emnist_28/", "./data/emnist_64/"]
        train_prefixs = ["./data/cifar10_july8_v3/", "./data/imagenet_may26_v3/", "./data/kws_chu_july19_v3/", "./data/super_v3/", "./data/emnist_28_v3/", "./data/emnist_64_v3/"]
        train_data = load_datas(train_prefixs, FLAGS.train_prefix, conditional=FLAGS.train_conditional)
    else:
        train_data = load_data(FLAGS.train_prefix, conditional=FLAGS.train_conditional)
    print("Done loading training data..")
    train(train_data)

if __name__ == '__main__':
    tf.app.run()

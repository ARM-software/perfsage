import tensorflow as tf
import numpy as np

import models as models
import layers as layers
from aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

class SupervisedPerfSAGE(models.SampleAndAggregate):
    """Implementation of supervised PerfSAGE."""

    def __init__(self, num_preds, 
            placeholders, category_count, category_embedding_dim, degrees,
            layer_infos, graph_node_max, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = placeholders["adj"]
        self.features = placeholders["features"]
        self.category_lookup = placeholders["category_lookup"]
        self.category_embedding_dim = category_embedding_dim
        self.category_count = category_count
        self.category_embeddings = tf.Variable(tf.random_normal([self.category_count+1, self.category_embedding_dim]), trainable=True)
        np_mask = np.vstack((np.ones((self.category_count, self.category_embedding_dim)), np.zeros((1, self.category_embedding_dim))))
        self.category_mask = tf.constant(np_mask, dtype=tf.float32)
        self.category_embeddings_mask = self.category_embeddings * self.category_mask
        self.degrees = degrees
        self.concat = concat
        self.num_preds = num_preds
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if self.features is None else int(self.features.shape[1])) + self.category_embedding_dim + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos
        self.graph_node_max = graph_node_max

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()


    def build(self):
        samples1, support_sizes1 = self.sample(self.inputs1, self.adj_info, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(samples1, self.features, self.category_lookup, self.category_embeddings_mask, self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size)
        dim_mult = 2 if self.concat else 1
        output1_dim = dim_mult*self.dims[-1]

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        self.outputs1 = tf.reshape(self.outputs1, (-1, self.graph_node_max, output1_dim))
        self.outputs1 = tf.reduce_mean(self.outputs1, 1)
        self.outputs1 = tf.reshape(self.outputs1, (-1, output1_dim))

        dim_mult = 2 if self.concat else 1
        self.node_pred = layers.Dense(output1_dim, self.num_preds, 
                dropout=self.placeholders['dropout'],
                act=lambda x : x)
        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)
        self.preds = self.predict()

        self._loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _loss(self):
        l2_param = FLAGS.weight_decay

        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += l2_param * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += l2_param * tf.nn.l2_loss(var)
        #self.loss += l2_param * tf.nn.l2_loss(self.category_embeddings)
       
        # prediction loss
        apes = tf.abs(self.preds - self.placeholders['preds']) / self.placeholders['preds']
        self.loss += tf.reduce_mean(apes)
        self.loss = tf.identity(self.loss, name="loss")

        tf.summary.scalar('loss', self.loss)

    def predict(self):
        return tf.identity(self.node_preds, name="predicts")

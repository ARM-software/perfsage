from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

np.random.seed(123)

class GraphMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, Gs, graph_dict, node_dict, node_count,  
            placeholders, features, category_lookup, prediction_dict, num_preds, 
            graph_node_max, batch_size=2, max_degree=25,
            **kwargs):

        self.Gs = Gs
        self.graph_dict = graph_dict
        self.node_dict = node_dict
        self.node_count = node_count
        self.placeholders = placeholders
        self.features = features
        self.category_lookup = category_lookup
        self.category_count = max(self.category_lookup) + 1
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.graph_node_max = graph_node_max
        self.batch_num = 0
        self.prediction_dict = prediction_dict
        self.num_preds = num_preds

        self.train_gids = self.graph_dict["train"]
        self.val_gids = self.graph_dict["val"]
        self.test_gids = self.graph_dict["test"]
        self.train_count = len(self.train_gids)
        self.graph_count = len(self.train_gids) + len(self.val_gids) + len(self.test_gids)

        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()

    def construct_adj(self):
        adj = self.node_count*np.ones((self.node_count+1, self.max_degree)).astype(int)
        deg = np.zeros((self.node_count,)).astype(int)

        for gid in self.train_gids:
            G = self.Gs[gid]
            for nodeid in G.nodes():
                neighbors = np.array([self.node_dict[gid][neighbor] for neighbor in G.neighbors(nodeid)])
                deg[self.node_dict[gid][nodeid]] = len(neighbors)
                if len(neighbors) == 0:
                    continue
                if len(neighbors) > self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                elif len(neighbors) < self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                adj[self.node_dict[gid][nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = self.node_count*np.ones((self.node_count+1, self.max_degree)).astype(int)

        for gid, G in enumerate(self.Gs):
            for nodeid in G.nodes():
                neighbors = np.array([self.node_dict[gid][neighbor] for neighbor in G.neighbors(nodeid)])
                if len(neighbors) == 0:
                    continue
                if len(neighbors) > self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                elif len(neighbors) < self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                adj[self.node_dict[gid][nodeid], :] = neighbors
        return adj

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_gids = np.random.permutation(self.train_gids)
        self.batch_num = 0

    def end(self):
        return self.batch_num * self.batch_size >= self.train_count

    def batch_feed_dict(self, batch_graphs, val=False, lite=True, load=False):
        graph_node_start = 1
        batch_nodes = np.zeros(len(batch_graphs)*self.graph_node_max).astype(int)
        batch_features = self.features[-1, :]
        batch_category_lookup = np.array([self.category_count])
        batch_adj = np.zeros(self.max_degree).astype(int)
        batch_node_lookup = np.zeros(self.node_count+1).astype(int)
        
        for idx, gid in enumerate(batch_graphs):
            graph_nodes = np.array(self.node_dict[gid]).astype(int)
            if lite: 
                graph_features = self.features[graph_nodes, :]
                batch_features = np.vstack((batch_features, graph_features))
                batch_category_lookup = np.hstack((batch_category_lookup, self.category_lookup[graph_nodes]))
                
                batch_graph_nodes = np.array(range(len(graph_nodes))) + graph_node_start
                batch_node_lookup[graph_nodes] = batch_graph_nodes
                batch_nodes[idx*self.graph_node_max:idx*self.graph_node_max+len(graph_nodes)] = batch_graph_nodes
                if val:
                    graph_adj = self.test_adj[graph_nodes, :].astype(int)
                else:
                    graph_adj = self.adj[graph_nodes, :].astype(int)
                batch_adj = np.vstack((batch_adj, batch_node_lookup[graph_adj]))
                graph_node_start += len(graph_nodes)
            else:
                nodes = np.ones(self.graph_node_max) * self.node_count
                nodes[:len(graph_nodes)] = graph_nodes
                batch_nodes[idx*self.graph_node_max:(idx+1)*self.graph_node_max] = nodes
                batch_features = self.features
                batch_category_lookup = self.category_lookup
                if val:
                    batch_adj = self.test_adj
                else:
                    batch_adj = self.adj
              
        preds = np.vstack([self.prediction_dict['inference_time'][gid] for gid in batch_graphs])
        feed_dict = {}

        feed_dict.update({self.placeholders['batch_size'] : len(batch_nodes)})
        feed_dict.update({self.placeholders['batch']: batch_nodes})
        feed_dict.update({self.placeholders['preds']: preds})
        feed_dict.update({self.placeholders['features']: batch_features})
        feed_dict.update({self.placeholders['category_lookup']: batch_category_lookup})
        feed_dict.update({self.placeholders['adj']: batch_adj})

        return feed_dict, preds

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, self.train_count)
        batch_graphs = self.train_gids[start_idx:end_idx]
        return self.batch_feed_dict(batch_graphs, val=False)

    def node_val_feed_dict(self, size=-1, test=False, load=False):
        if test:
            val_ids = self.test_gids
        else:
            val_ids = self.val_gids
        if not size is -1:
            val_ids = np.random.choice(val_ids, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_ids, val=True, load=load)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False, load=False):
        if test:
            val_gids = self.test_gids
        elif load:
            #val_gids = self.train_gids + self.val_gids + self.test_gid
            val_gids = range(self.graph_count)
        else:
            val_gids = self.val_gids
        val_gids_subset = val_gids[iter_num*size:min((iter_num+1)*size, len(val_gids))]
    
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_gids_subset, val=True, load=load)
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_gids)

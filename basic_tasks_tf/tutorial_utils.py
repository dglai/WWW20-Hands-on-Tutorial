import pandas as pd
import os

def setup_tf():
    os.environ['USE_OFFICIAL_TFDLPACK']='true'
    os.environ['DGLBACKEND']='tensorflow'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def load_zachery():
    import dgl
    import numpy as np    
    import tensorflow as tf
    nodes_data = pd.read_csv('data/nodes.csv')
    edges_data = pd.read_csv('data/edges.csv')
    src = edges_data['Src'].to_numpy()
    dst = edges_data['Dst'].to_numpy()
    g = dgl.graph((src, dst))
    club = nodes_data['Club'].to_list()
    # Convert to categorical integer values with 0 for 'Mr. Hi', 1 for 'Officer'.
    club = tf.constant([c == 'Officer' for c in club], dtype=tf.int64)
    # We can also convert it to one-hot encoding.
    club_onehot = tf.one_hot(club, np.max(club) + 1)
    g.ndata.update({'club' : club, 'club_onehot' : club_onehot})
    return g

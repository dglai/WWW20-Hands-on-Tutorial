import dgl
import pandas as pd
import torch
import torch.nn.functional as F

def load_zachery():
    nodes_data = pd.read_csv('data/nodes.csv')
    edges_data = pd.read_csv('data/edges.csv')
    src = edges_data['Src'].to_numpy()
    dst = edges_data['Dst'].to_numpy()
    g = dgl.graph((src, dst))
    club = nodes_data['Club'].to_list()
    # Convert to categorical integer values with 0 for 'Mr. Hi', 1 for 'Officer'.
    club = torch.tensor([c == 'Officer' for c in club]).long()
    # We can also convert it to one-hot encoding.
    club_onehot = F.one_hot(club)
    g.ndata.update({'club' : club, 'club_onehot' : club_onehot})
    return g

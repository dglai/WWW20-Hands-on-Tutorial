import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph

# Load Pytorch as backend
dgl.load_backend('pytorch')

import numpy as np
import pandas as pd
from scipy import stats
from scipy import sparse as spsp

import pickle
user_movie_spm = pickle.load(open('bx/bx_train.pkl', 'rb'))
abstracts = pickle.load(open('bx/bx_book_abstract.pkl', 'rb')).asnumpy()
titles = pickle.load(open('bx/bx_book_title.pkl', 'rb')).asnumpy()
features = torch.tensor(np.concatenate((titles, abstracts), 1), dtype=torch.float32)
valid_set, test_set = pickle.load(open('bx/bx_eval.pkl', 'rb'))
neg_valid, neg_test = pickle.load(open('bx/bx_neg.pkl', 'rb'))

num_users = user_movie_spm.shape[0]
num_movies = user_movie_spm.shape[1]

users_valid = np.arange(num_users)
movies_valid = valid_set
users_test = np.arange(num_users)
movies_test = test_set

#features = torch.cat([features, movie_popularity, v], 1)
#one_hot = torch.tensor(np.diag(np.ones(shape=(num_movies))), dtype=torch.float32)
#features = torch.cat([features, one_hot], 1)
in_feats = features.shape[1]
print('#feats:', in_feats)

user_deg = user_movie_spm.dot(np.ones((num_movies)))
print(len(user_deg))
user_deg1 = np.nonzero(user_deg == 1)[0]
user_deg2 = np.nonzero(user_deg == 2)[0]
user_deg3 = np.nonzero(user_deg == 3)[0]
user_deg4 = np.nonzero(user_deg == 4)[0]
user_deg5 = np.nonzero(user_deg == 5)[0]
user_deg6 = np.nonzero(user_deg == 6)[0]
user_deg7 = np.nonzero(user_deg == 7)[0]
user_deg8 = np.nonzero(user_deg == 8)[0]
user_deg9 = np.nonzero(user_deg == 9)[0]
user_deg10 = np.nonzero(np.logical_and(10 <= user_deg, user_deg < 20))[0]
user_deg20 = np.nonzero(np.logical_and(20 <= user_deg, user_deg < 30))[0]
user_deg30 = np.nonzero(np.logical_and(30 <= user_deg, user_deg < 40))[0]
user_deg40 = np.nonzero(np.logical_and(40 <= user_deg, user_deg < 50))[0]
user_deg50 = np.nonzero(np.logical_and(50 <= user_deg, user_deg < 60))[0]
user_deg60 = np.nonzero(np.logical_and(60 <= user_deg, user_deg < 70))[0]
user_deg70 = np.nonzero(np.logical_and(70 <= user_deg, user_deg < 80))[0]
user_deg80 = np.nonzero(np.logical_and(80 <= user_deg, user_deg < 90))[0]
user_deg90 = np.nonzero(np.logical_and(90 <= user_deg, user_deg < 100))[0]
user_deg100 = np.nonzero(user_deg >= 100)[0]
print(len(user_deg1))
print(len(user_deg2))
print(len(user_deg3))
print(len(user_deg4))
print(len(user_deg5))
print(len(user_deg6))
print(len(user_deg7))
print(len(user_deg8))
print(len(user_deg9))
print(len(user_deg10))
print(len(user_deg20))
print(len(user_deg30))
print(len(user_deg40))
print(len(user_deg50))
print(len(user_deg60))
print(len(user_deg70))
print(len(user_deg80))
print(len(user_deg90))
print(len(user_deg100))

movie_deg = user_movie_spm.transpose().dot(np.ones((num_users)))
test_deg = np.zeros((num_users))
for i in range(num_users):
    movie = int(movies_test[i])
    test_deg[i] = movie_deg[movie]
test_deg_dict = {}
for i in range(1, 10):
    test_deg_dict[i] = np.nonzero(test_deg == i)[0]
for i in range(1, 10):
    test_deg_dict[i*10] = np.nonzero(np.logical_and(i*10 <= test_deg, test_deg < (i+1)*10))[0]
test_deg_dict[100] = np.nonzero(test_deg >= 100)[0]
tmp = 0
for key, deg in test_deg_dict.items():
    print(key, len(deg))
    tmp += len(deg)
print(num_users, tmp)

from SLIM import SLIM, SLIMatrix
model = SLIM()
params = {'algo': 'cd', 'nthreads': 16, 'l1r': 2, 'l2r': 1}
trainmat = SLIMatrix(user_movie_spm.tocsr())
model.train(params, trainmat)
model.save_model(modelfname='slim_model.csr', mapfname='slim_map.csr')

from slim_load import read_csr

movie_spm = read_csr('slim_model.csr')
print('#edges:', movie_spm.nnz)
print('most similar:', np.max(movie_spm.data))
print('most unsimilar:', np.min(movie_spm.data))

deg = movie_spm.dot(np.ones((num_movies)))
print(np.sum(deg == 0))
print(len(deg))
print(movie_spm.sum(0))

g = dgl.DGLGraph(movie_spm, readonly=True)
g.edata['similarity'] = torch.tensor(movie_spm.data, dtype=torch.float32)

user_id = user_movie_spm.row
movie_id = user_movie_spm.col
movie_deg = user_movie_spm.transpose().dot(np.ones((num_users,)))
movie_ratio = movie_deg / np.sum(movie_deg)
# 1e-6 is a hyperparameter for this dataset.
movie_sample_prob = 1 - np.maximum(1 - np.sqrt(1e-5 / movie_ratio), 0)
sample_prob = movie_sample_prob[movie_id]
sample = np.random.uniform(size=(len(movie_id),))
user_id = user_id[sample_prob > sample]
movie_id = movie_id[sample_prob > sample]
print('#samples:', len(user_id))
spm = spsp.coo_matrix((np.ones((len(user_id),)), (user_id, movie_id)))
print(spm.shape)
movie_deg = spm.transpose().dot(np.ones((num_users,)))
print(np.sum(movie_deg == 0))

movie_spm = np.dot(spm.transpose(), spm)
print(movie_spm.nnz)
dense_movie = np.sort(movie_spm.todense())
topk_movie = dense_movie[:,-50]
topk_movie_spm = movie_spm > topk_movie

from sklearn.metrics.pairwise import cosine_similarity
movie_spm = cosine_similarity(spm.transpose(),dense_output=False)

dense_movie = np.sort(movie_spm.todense())
topk_movie = dense_movie[:,-20]
topk_movie_spm = movie_spm > topk_movie

movie_spm = spsp.csr_matrix(topk_movie_spm)
print(movie_spm.nnz)
g = dgl.DGLGraph(movie_spm, readonly=True)

#from sageconv import SAGEConv
from dgl.nn.pytorch import conv as dgl_conv

class GraphSAGEModel(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGEModel, self).__init__()
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(dgl_conv.SAGEConv(in_feats, n_hidden, aggregator_type,
                                        feat_drop=dropout, activation=None))
        elif n_layers > 1:
            # input layer
            self.layers.append(dgl_conv.SAGEConv(in_feats, n_hidden, aggregator_type,
                                        feat_drop=dropout, activation=activation))
            # hidden layer
            for i in range(n_layers - 2):
                self.layers.append(dgl_conv.SAGEConv(n_hidden, n_hidden, aggregator_type,
                                            feat_drop=dropout, activation=activation))
            # output layer
            self.layers.append(dgl_conv.SAGEConv(n_hidden, out_dim, aggregator_type,
                                        feat_drop=dropout, activation=None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
            #h = layer(g, prev_h, g.edata['similarity'])
            #h = tmp + prev_h
            #prev_h = h
        return h

class EncodeLayer(nn.Module):
    def __init__(self, in_feats, num_hidden, device):
        super(EncodeLayer, self).__init__()
        self.proj = nn.Linear(in_feats, int(num_hidden))
        #self.emb = nn.Embedding(num_movies, int(num_hidden))
        #self.nid = torch.arange(num_movies).to(device)

    def forward(self, feats):
        #return torch.cat([self.proj(feats), self.emb(self.nid)], 1)
        return self.proj(feats)
        #return self.emb(self.nid)

beta = 0
gamma = 0

class FISM(nn.Module):
    def __init__(self, user_movie_spm, gconv_p, gconv_q, in_feats, num_hidden, device):
        super(FISM, self).__init__()
        self.num_users = user_movie_spm.shape[0]
        self.num_movies = user_movie_spm.shape[1]
        self.b_u = nn.Parameter(torch.zeros(num_users))
        self.b_i = nn.Parameter(torch.zeros(num_movies))
        self.user_deg = torch.tensor(user_movie_spm.dot(np.ones(num_movies)), dtype=torch.float32).to(device)
        values = user_movie_spm.data
        indices = np.vstack((user_movie_spm.row, user_movie_spm.col))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(values)
        self.user_item_spm = torch.sparse_coo_tensor(indices, values, user_movie_spm.shape).to(device)
        self.users = user_movie_spm.row
        self.movies = user_movie_spm.col
        self.ratings = user_movie_spm.data
        user_movie_csr = user_movie_spm.tocsr()
        self.neg_train = [
                np.setdiff1d(np.arange(self.num_movies),
                    user_movie_csr[i].nonzero()[1])
                for i in range(self.num_users)
                ]
        self.encode_p = EncodeLayer(in_feats, num_hidden, device)
        self.encode_q = EncodeLayer(in_feats, num_hidden, device)
        self.gconv_p = gconv_p
        self.gconv_q = gconv_q

    def _est_rating(self, P, Q, user_idx, item_idx):
        bu = self.b_u[user_idx]
        bi = self.b_i[item_idx]
        user_emb = torch.sparse.mm(self.user_item_spm, P)
        user_emb = (user_emb[user_idx] - P[item_idx]) / \
                (torch.unsqueeze(self.user_deg[user_idx], 1) - 1).clamp(min=1)
        tmp = torch.mul(user_emb, Q[item_idx])
        r_ui = bu + bi + torch.sum(tmp, 1)
        return r_ui
    
    def est_rating(self, g, features, user_idx, item_idx, neg_item_idx):
        P = self.gconv_p(g, self.encode_p(features))
        Q = self.gconv_q(g, self.encode_q(features))
        r = self._est_rating(P, Q, user_idx, item_idx)
        neg_sample_size = len(neg_item_idx) / len(user_idx)
        neg_r = self._est_rating(P, Q, np.repeat(user_idx, neg_sample_size), neg_item_idx)
        return torch.unsqueeze(r, 1), neg_r.reshape((-1, int(neg_sample_size)))

    def loss(self, P, Q, r_ui, neg_r_ui):
        diff = 1 - (r_ui - neg_r_ui)
        return torch.sum(torch.mul(diff, diff)/2) \
            + beta/2 * torch.sum(torch.mul(P, P) + torch.mul(Q, Q)) \
            + gamma/2 * (torch.sum(torch.mul(self.b_u, self.b_u)) + torch.sum(torch.mul(self.b_i, self.b_i)))

    def forward(self, g, features, neg_sample_size):
        P = self.gconv_p(g, self.encode_p(features))
        Q = self.gconv_q(g, self.encode_q(features))
        tot = len(self.users)
        pos_idx = np.random.choice(tot, 1024)
        user_idx = self.users[pos_idx]
        item_idx = self.movies[pos_idx]
        neg_item_idx = np.array([
            np.random.choice(self.neg_train[i], neg_sample_size)
            for i in user_idx
            ]).flatten()
        neg_item_idx = np.random.choice(self.num_movies, len(pos_idx) * neg_sample_size)
        r_ui = self._est_rating(P, Q, user_idx, item_idx)
        neg_r_ui = self._est_rating(P, Q, np.repeat(user_idx, neg_sample_size), neg_item_idx)
        r_ui = torch.unsqueeze(r_ui, 1)
        neg_r_ui = neg_r_ui.reshape((-1, int(neg_sample_size)))
        return self.loss(P, Q, r_ui, neg_r_ui)

def RecValid(model, g, features):
    model.eval()
    with torch.no_grad():
        neg_movies_eval = neg_valid[users_valid]
        r, neg_r = model.est_rating(g, features, users_valid, movies_valid, neg_movies_eval.flatten())
        hits10 = (torch.sum(neg_r > r, 1) <= 10).cpu().numpy()
        return np.mean(hits10)
    
def RecTest(model, g, features):
    model.eval()
    with torch.no_grad():
        neg_movies_eval = neg_test[users_test]
        r, neg_r = model.est_rating(g, features, users_test, movies_test, neg_movies_eval.flatten())
        hits10 = (torch.sum(neg_r > r, 1) <= 10).cpu().numpy()
        for popularity, users in test_deg_dict.items():
            print(popularity, np.mean(hits10[users]))
        return np.mean(hits10)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


#Model hyperparameters
n_hidden = 16
n_layers = 0
dropout = 0
aggregator_type = 'gcn'

# create GraphSAGE model
gconv_p = GraphSAGEModel(n_hidden,
                         n_hidden,
                         n_hidden,
                         n_layers,
                         F.relu,
                         dropout,
                         aggregator_type)

gconv_q = GraphSAGEModel(n_hidden,
                         n_hidden,
                         n_hidden,
                         n_layers,
                         F.relu,
                         dropout,
                         aggregator_type)

model = FISM(user_movie_spm, gconv_p, gconv_q, in_feats, n_hidden, device).to(device)
g.to(device)
features = features.to(device)

# Training hyperparameters
weight_decay = 1e-3
n_epochs = 30000
lr = 3e-5
neg_sample_size = 20

# use optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# initialize graph
dur = []
prev_acc = 0
for epoch in range(n_epochs):
    model.train()
    loss = model(g, features, neg_sample_size)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        hits10 = RecValid(model, g, features)
        print("Epoch {:05d} | Loss {:.4f} | HITS@10:{:.4f}".format(epoch, loss.item(), np.mean(hits10)))

print()
# Let's save the trained node embeddings.
hits10 = RecTest(model, g, features)
print("Test HITS@10:{:.4f}".format(np.mean(hits10)))

# use optimizer
lr = 1e-4
weight_decay = 1e-2

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# initialize graph
dur = []
prev_acc = 0
for epoch in range(1000):
    model.train()
    loss = model(g, features, neg_sample_size)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        hits10 = RecValid(model, g, features)
        print("Epoch {:05d} | Loss {:.4f} | HITS@10:{:.4f}".format(epoch, loss.item(), np.mean(hits10)))

print()
# Let's save the trained node embeddings.
hits10 = RecTest(model, g, features)
print("Test HITS@10:{:.4f}".format(np.mean(hits10)))

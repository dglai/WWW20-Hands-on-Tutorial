import networkx as nx
import torch
import scipy.sparse as sp

g = nx.karate_club_graph().to_undirected().to_directed()
src = []
dst = []
for u, v in g.edges():
    src.append(u)
    dst.append(v)

with open('edges.txt', 'w') as f:
    for u, v in zip(src, dst):
        f.write('{} {}\n'.format(u, v))

torch.save(torch.tensor(src), 'src.pt')
torch.save(torch.tensor(dst), 'dst.pt')

spmat = nx.to_scipy_sparse_matrix(g)
print(spmat)
sp.save_npz('scipy_adj.npz', spmat)

from networkx.readwrite import json_graph
import json

with open('adj.json', 'w') as f:
    json.dump(json_graph.adjacency_data(g), f)

node_feat = torch.randn((34, 5)) / 10.
edge_feat = torch.ones((156,))
torch.save(node_feat, 'node_feat.pt')
torch.save(edge_feat, 'edge_feat.pt')

import numpy as np
from scipy import sparse as spsp

def read_csr(filename):
    f = open(filename, 'r')
    all_rows = []
    all_cols = []
    all_vals = []
    for i, line in enumerate(f.readlines()):
        strs = line.split(' ')
        cols = [int(s) for s in strs[1::2]]
        vals = [float(s) for s in strs[2::2]]
        all_cols.extend(cols)
        all_vals.extend(vals)
        all_rows.extend([i for _ in cols])
    all_rows = np.array(all_rows, dtype=np.int64)
    all_cols = np.array(all_cols, dtype=np.int64)
    all_vals = np.array(all_vals, dtype=np.float32)
    mat = spsp.coo_matrix((all_vals, (all_rows, all_cols)))
    return mat


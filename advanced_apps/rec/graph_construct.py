import numpy as np
from scipy import sparse as spsp

from SLIM import SLIM, SLIMatrix
from sklearn.metrics.pairwise import cosine_similarity

def eval_SLIM(model, trainmat, items_test, neg_test):
    candidates = {}
    num_users = neg_test.shape[0]
    neg_sample_size = neg_test.shape[1]
    for i, neg_items in enumerate(neg_test):
        candidates[i] = np.zeros(shape=(neg_items.shape[0] + 1,), dtype=np.int64)
        candidates[i][:neg_sample_size] = neg_items
        candidates[i][-1] = int(items_test[i])
    rcmd_list = model.predict(trainmat, nrcmds=10, negitems=candidates, nnegs=100)
    rcmd_mat = np.ones(shape=(num_users, len(rcmd_list[0])), dtype=np.int64)
    for i, rcmd in rcmd_list.items():
        rcmd_mat[i] = rcmd

    def test(rcmd_mat, true_list):
        hits = 0
        true_list = np.expand_dims(true_list, 1)
        hits = np.sum(rcmd_mat == true_list)
        print("Number of test(valid) users: %d(%d), hits@%d: %.4f" % 
              (len(true_list), len(rcmd_list), len(rcmd_list[0]),
               hits / len(rcmd_list)))
        return hits / len(rcmd_list)
    test(rcmd_mat, items_test)


def create_SLIM_graph(user_item_spm, l1r=1, l2r=1, test=False, test_set=None):
    model = SLIM()
    params = {'algo': 'cd', 'nthreads': 32, 'l1r': l1r, 'l2r': l2r}
    trainmat = SLIMatrix(user_item_spm.tocsr())
    model.train(params, trainmat)
    item_spm = model.to_csr()
    if test and test_set is not None:
        users_test, items_test, neg_test = test_set
        eval_SLIM(model, trainmat, items_test, neg_test)
    return item_spm

def create_cooccur_graph(user_item_spm, downsample_factor=1e-5, topk=50):
    num_users = user_item_spm.shape[0]
    user_item_spm = user_item_spm.tocoo()
    user_id = user_item_spm.row
    item_id = user_item_spm.col
    item_deg = user_item_spm.transpose().dot(np.ones((num_users,)))
    item_ratio = item_deg / np.sum(item_deg)
    item_ratio[item_ratio == 0] = 1
    # 1e-6 is a hyperparameter for this dataset.
    item_sample_prob = 1 - np.maximum(1 - np.sqrt(downsample_factor / item_ratio), 0)
    sample_prob = item_sample_prob[item_id]
    sample = np.random.uniform(size=(len(item_id),))
    user_id = user_id[sample_prob > sample]
    item_id = item_id[sample_prob > sample]
    spm = spsp.coo_matrix((np.ones((len(user_id),)), (user_id, item_id)))
    item_deg = spm.transpose().dot(np.ones((num_users,)))

    item_spm = np.dot(spm.transpose(), spm)
    if topk is not None:
        dense_item = np.sort(item_spm.todense())
        topk_item = dense_item[:,-topk]
        topk_item_spm = item_spm > topk_item
        topk_item_spm = spsp.csr_matrix(topk_item_spm)
        return topk_item_spm
    else:
        return item_spm

def create_cosine_graph(user_item_spm, topk=None):
    item_spm = cosine_similarity(user_item_spm.transpose(),dense_output=False)
    if topk is not None:
        dense_item = np.sort(item_spm.todense())
        topk_item = dense_item[:,-topk]
        topk_item_spm = item_spm > topk_item
        topk_item_spm = spsp.csr_matrix(topk_item_spm)
        topk_item_spm = item_spm.multiply(topk_item_spm)
        return topk_item_spm
    else:
        return item_spm

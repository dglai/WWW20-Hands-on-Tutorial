import pickle
import numpy as np
from scipy import sparse as spsp

def load_movielens():
    user_movie_spm = pickle.load(open('movielens/movielens_orig_train.pkl', 'rb'))
    features = pickle.load(open('movielens/movielens_features.pkl', 'rb'))
    valid_set, test_set = pickle.load(open('movielens/movielens_eval.pkl', 'rb'))
    neg_valid, neg_test = pickle.load(open('movielens/movielens_neg.pkl', 'rb'))

    num_users = user_movie_spm.shape[0]
    num_movies = user_movie_spm.shape[1]

    users_valid = np.arange(num_users)
    movies_valid = valid_set
    users_test = np.arange(num_users)
    movies_test = test_set

    return user_movie_spm, features, \
            (users_valid, movies_valid), \
            (users_test, movies_test), \
            neg_valid, neg_test


def load_bookcrossing():
    user_book_spm = pickle.load(open('bx/bx_train.pkl', 'rb'))
    abstracts = pickle.load(open('bx/bx_book_abstract.pkl', 'rb'))
    titles = pickle.load(open('bx/bx_book_title.pkl', 'rb'))
    features = np.concatenate((titles, abstracts), 1)
    valid_set, test_set = pickle.load(open('bx/bx_eval.pkl', 'rb'))
    neg_valid, neg_test = pickle.load(open('bx/bx_neg.pkl', 'rb'))

    num_users = user_book_spm.shape[0]
    num_books = user_book_spm.shape[1]

    users_valid = np.arange(num_users, dtype=np.int64)
    books_valid = valid_set
    users_test = np.arange(num_users, dtype=np.int64)
    books_test = test_set

    return user_book_spm, features, \
            (users_valid, books_valid), \
            (users_test, books_test), \
            neg_valid, neg_test

def load_yelp2018():
    user_item_spm = pickle.load(open('yelp/yelp2018_orig_train.pkl', 'rb'))
    features = pickle.load(open('yelp/yelp2018_entity_embed_features.pkl', 'rb'))
    valid_set, test_set = pickle.load(open('yelp/yelp2018_eval.pkl', 'rb'))
    neg_valid, neg_test = pickle.load(open('yelp/yelp2018_neg.pkl', 'rb'))

    num_users = user_item_spm.shape[0]
    num_items = user_item_spm.shape[1]

    users_valid = np.arange(num_users, dtype=np.int64)
    items_valid = valid_set
    users_test = np.arange(num_users, dtype=np.int64)
    items_test = test_set

    return user_item_spm, features, \
            (users_valid, items_valid), \
            (users_test, items_test), \
            neg_valid, neg_test

def load_data(name):
    if name == 'movielens':
        return load_movielens()
    elif name == 'bx':
        return load_bookcrossing()
    elif name == 'yelp':
        return load_yelp2018()
    else:
        raise Exception('We only support movielens, bx and yelp')

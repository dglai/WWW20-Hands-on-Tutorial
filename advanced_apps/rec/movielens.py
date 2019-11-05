
import pandas as pd
import dgl
import os
import torch
import numpy as np
import scipy.sparse as sp
import time
from functools import partial
import stanfordnlp
import re
import tqdm
import string

def _download_and_extract(url, path, filename):
    import zipfile, tarfile
    import requests

    fn = os.path.join(path, filename)

    while True:
        try:
            tar = tarfile.open(fn, 'r:gz')
            tar.extractall()
            tar.close()
            break
        except Exception:
            os.makedirs(path, exist_ok=True)
            f_remote = requests.get(url, stream=True)
            sz = f_remote.headers.get('content-length')
            assert f_remote.status_code == 200, 'fail to open {}'.format(url)
            with open(fn, 'wb') as writer:
                for chunk in f_remote.iter_content(chunk_size=1024*1024):
                    writer.write(chunk)
            print('Download finished. Unzipping the file...')

class MovieLens(object):
    def __init__(self, directory, neg_size=99):
        '''
        directory: path to movielens directory which should have the three
                   files:
                   users.dat
                   movies.dat
                   ratings.dat
        '''
        url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/movielens.tar.gz'
        if not os.path.exists(os.path.join(directory, 'users.dat')):
            print('File not found. Downloading from', url)
            _download_and_extract(url, directory, 'movielens.tar.gz')
        directory = os.path.join(directory, 'movielens')

        users = []
        movies = []
        ratings = []

        # read users
        with open(os.path.join(directory, 'users.dat')) as f:
            for l in f:
                id_, gender, age, occupation, zip_ = l.strip().split('::')
                users.append({
                    'id': int(id_),
                    'gender': gender,
                    'age': age,
                    'occupation': occupation,
                    'zip': zip_,
                    })
        self.users = pd.DataFrame(users)

        # read movies
        with open(os.path.join(directory, 'movies.dat'), encoding='latin1') as f:
            for l in f:
                id_, title, genres = l.strip().split('::')
                genres_set = set(genres.split('|'))

                # extract year
                assert re.match(r'.*\([0-9]{4}\)$', title)
                year = title[-5:-1]
                title = title[:-6].strip()

                data = {'id': int(id_), 'title': title, 'year': year}
                for g in genres_set:
                    data[g] = True
                movies.append(data)
        self.movies = (
                pd.DataFrame(movies)
                .fillna(False)
                .astype({'year': 'category'}))
        self.genres = self.movies.columns[self.movies.dtypes == bool]

        # read ratings
        with open(os.path.join(directory, 'ratings.dat')) as f:
            for l in f:
                user_id, movie_id, rating, timestamp = [int(_) for _ in l.split('::')]
                ratings.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'timestamp': timestamp,
                    })
        ratings = pd.DataFrame(ratings)
        movie_count = ratings['movie_id'].value_counts()
        movie_count.name = 'movie_count'
        ratings = ratings.join(movie_count, on='movie_id')
        self.ratings = ratings

        # determine test and validation set
        self.ratings['timerank'] = self.ratings.groupby('user_id')['timestamp'].rank().astype('int')
        self.ratings['test_mask'] = (self.ratings['timerank'] == 1)
        self.ratings['valid_mask'] = (self.ratings['timerank'] == 2)

        # remove movies that only appear in validation and test set
        movies_selected = self.ratings[self.ratings['timerank'] > 2]['movie_id'].unique()
        self.ratings = self.ratings[self.ratings['movie_id'].isin(movies_selected)].copy()

        # drop users and movies which do not exist in ratings
        self.users = self.users[self.users['id'].isin(self.ratings['user_id'])]
        self.movies = self.movies[self.movies['id'].isin(self.ratings['movie_id'])]
        self.user_ids_invmap = {u: i for i, u in enumerate(self.users['id'])}
        self.movie_ids_invmap = {m: i for i, m in enumerate(self.movies['id'])}

        self.ratings['user_idx'] = self.ratings['user_id'].apply(lambda x: self.user_ids_invmap[x])
        self.ratings['movie_idx'] = self.ratings['movie_id'].apply(lambda x: self.movie_ids_invmap[x])

        # parse movie features
        movie_data = {}
        num_movies = len(self.movies)
        num_users = len(self.users)

        movie_genres = torch.from_numpy(self.movies[self.genres].values.astype('float32'))
        movie_data['genre'] = movie_genres
        movie_data['year'] = \
                torch.LongTensor(self.movies['year'].cat.codes.values.astype('int64') + 1)

        nlp = stanfordnlp.Pipeline(use_gpu=False, processors='tokenize,lemma')
        vocab = set()
        title_words = []
        for t in tqdm.tqdm(self.movies['title'].values):
            doc = nlp(t)
            words = set()
            for s in doc.sentences:
                words.update(w.lemma.lower() for w in s.words
                             if not re.fullmatch(r'['+string.punctuation+']+', w.lemma))
            vocab.update(words)
            title_words.append(words)
        vocab = list(vocab)
        vocab_invmap = {w: i for i, w in enumerate(vocab)}
        # bag-of-words
        movie_data['title'] = torch.zeros(num_movies, len(vocab))
        for i, tw in enumerate(tqdm.tqdm(title_words)):
            movie_data['title'][i, [vocab_invmap[w] for w in tw]] = 1
        self.vocab = vocab
        self.vocab_invmap = vocab_invmap

        self.movie_data = movie_data

        # unobserved items for each user in training set
        self.neg_train = [None] * len(self.users)
        # negative examples for validation and test for evaluating ranking
        self.neg_valid = np.zeros((len(self.users), neg_size), dtype='int64')
        self.neg_test = np.zeros((len(self.users), neg_size), dtype='int64')
        rating_groups = self.ratings.groupby('user_idx')

        for u in range(len(self.users)):
            interacted_movies = self.ratings['movie_idx'][rating_groups.indices[u]]
            timerank = self.ratings['timerank'][rating_groups.indices[u]]

            interacted_movies_valid = interacted_movies[timerank > 2]
            neg_samples = np.setdiff1d(np.arange(len(self.movies)), interacted_movies_valid)
            self.neg_train[u] = neg_samples
            self.neg_valid[u] = np.random.choice(neg_samples, neg_size)

            interacted_movies_test = interacted_movies[timerank > 1]
            neg_samples = np.setdiff1d(np.arange(len(self.movies)), interacted_movies_test)
            self.neg_test[u] = np.random.choice(neg_samples, neg_size)

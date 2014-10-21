# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

from __future__ import print_function

from pprint import pprint
from time import time
import logging
import json
import itertools as it

from argparse import ArgumentParser
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer

from utils.lfcorpus import get_data_frame
from lflearn.feature_extract import TextExtractor, FeatureLang, \
    LengthVectorizer, FeaturePipeline

import nltk
nltk.data.path.append('./corpora/nltk_data/')

op = ArgumentParser()
op.add_argument("--scoring", type=str,
                help="data directory")
op.add_argument("--data_dir", type=str,
                help="data directory")
op.add_argument("--output_basename", type=str, default='Untitled',
                help="basename of file containing json dict output for combining")

op.add_argument('--shard', type=int, default=0,
                help="which split of the data are we")
op.add_argument('--n_shards', type=int, default=1,
                help="number of data splits")

args = op.parse_args()

assert 0 <= args.shard < args.n_shards


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

if args.data_dir is None:
    # Load 20 newsgroups corpus
    from sklearn.datasets import fetch_20newsgroups
    categories = [
        'alt.atheism',
        'talk.religion.misc'
    ]
    data = fetch_20newsgroups(subset='train', categories=categories)
else:
    # Load custom corpus
    data = get_data_frame(
        args.data_dir,
        lambda line: json.loads(line))
    categories = data.target_names

print("%d documents" % len(data.filenames))
print("%d categories" % len(categories))
print()

###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier


vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.3,
                             stop_words='english')

content_pipeline = FeaturePipeline([
    ('cont1', TextExtractor('content')),
    ('vec', vectorizer),
])
# pca_pipeline = PCAPipeline([
#     ('cont2', TextExtractor('content')),
#     ('vectf', TfidfVectorizer(sublinear_tf=True, max_df=0.4,
#                               stop_words='english')),
#     ('pca', TruncatedSVD(n_components=PCA_components))
# ])
lang_pipeline = FeaturePipeline([
    ('cont3', TextExtractor('content')),
    ('lang', FeatureLang()),
    ('dvec', DictVectorizer()),
])
len_pipeline = FeaturePipeline([
    ('cont4', TextExtractor('content')),
    ('len', LengthVectorizer())
])
preprocess = FeatureUnion([
    ('cp', content_pipeline),
    #('lp', lang_pipeline),
    # ('mp', len_pipeline)
])

# pipeline = Pipeline([
#     ('vect', CountVectorizer(lowercase=True, stop_words="english", max_df=0.30)),
#     ('tfidf', TfidfTransformer(sublinear_tf=True)),
#     ('norm', Normalizer()),
#     ('clf', SGDClassifier(penalty='elasticnet', n_iter=50)),
# ])

pipeline = Pipeline([
    ('fp', preprocess),
    ('norm', Normalizer()),
    ('clf', SGDClassifier(alpha=1e-4, l1_ratio=0.10, penalty='elasticnet', n_iter=50)),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way

# 'vect__max_df': (0.20, 0.30, 0.40),
# 'vect__lowercase': (True, False),
# 'vect__stop_words': ('english', None),
# 'vect__max_features': (None, 5000, 10000, 50000),
# 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
# 'tfidf__sublinear_tf': (True, False),
# 'tfidf__use_idf': (True, False),
# 'tfidf__norm': ('l1', 'l2'),
# 'clf__n_iter': (10, 50, 80),

param_grid = [
    {'fp__cp__vec__max_df': [0.1, 0.2, 0.3],
     'clf__loss': ['hinge'],
     'clf__alpha': [1e-2, 1e-3, 1e-4],
     'clf__l1_ratio': [0.00, 0.10, 0.20, 0.30]},

    # {'clf__loss': ['log'],
    # 'clf__alpha': [ 1e-4, 1e-6 ],
    # 'clf__l1_ratio': [ 0.40, 0.60 ]}
]

def one_from_each_list(lists):
    """list of lists -> all combinations of one from each list."""
    if not all(lists):
        return

    tuples = map(tuple, lists)
    max_indexes = map(lambda aa: len(aa) - 1, tuples)
    indexes = [0] * len(tuples)
    while True:
        yield map(lambda (aa, i): aa[i], zip(tuples, indexes))
        ok = False
        for pos in range(len(indexes)):
            if indexes[pos] == max_indexes[pos]:
                indexes[pos] = 0
            else:
                indexes[pos] += 1
                ok = True
                break
        if not ok:
            break

def flatten_param_grid_dict(d):
    """list of (k -> vv) -> equivalent list of (k -> v)."""
    keys = d.keys()
    values = d.values()
    for new_values in one_from_each_list(values):
        new_d = dict(zip(keys, map(lambda v: [v], new_values)))
        yield new_d

def get_param_grid_shard(param_grid, shard, n_shards):
    """get a new param grid that contains one shard's worth of work."""
    dd = []
    for d in param_grid:
        dd.extend(flatten_param_grid_dict(d))
    min_x = len(dd) * shard / n_shards
    max_x = len(dd) * (shard + 1) / n_shards
    print('selecting subset: [%d : %d) of %d' % (min_x, max_x, len(dd)))
    sub_dd = dd[min_x:max_x]
    return sub_dd

param_grid = get_param_grid_shard(param_grid, args.shard, args.n_shards)

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=1,
                               scoring=args.scoring)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(param_grid)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    parameters_tested = set(it.chain(*(i.keys() for i in param_grid)))
    bp = []
    for param_name in sorted(parameters_tested):
        k = param_name
        v = best_parameters[param_name]
        print("\t%s: %r" % (param_name, v))
        bp.append((k, v))

    d = {
        'best_score': grid_search.best_score_.astype(float),
        'best_parameters': bp,
    }
    f = '%s.%d.txt' % (args.output_basename, args.shard)
    s = json.dumps(d)
    open(f, 'w').write(s)

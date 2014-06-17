# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

from __future__ import print_function

from pprint import pprint
from time import time
import logging
import sys
import os
import json

import numpy as np
from omnihack import enumerator
from optparse import OptionParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.datasets.base import Bunch

op = OptionParser()
op.add_option("--data_dir", type=str,
              help="data directory")

(opts, args) = op.parse_args()
if opts.data_dir is None:
    op.error('Data directory not given')

if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def get_files(dirname, extension=".txt"):
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            if fname.endswith(extension):
                yield os.path.join(root, fname)


def get_category(fname):
    return os.path.basename(os.path.splitext(fname)[0])


def get_data_frame(dirname, get_data):
    category_codes = enumerator()
    x_nested = []
    y_nested = []
    fnames = [fn for fn in get_files(dirname)]
    num_files = len(fnames)
    max_num_lines = 0
    for fname in fnames:
        file_data = []
        num_lines = 0
        with open(fname) as f:
            for line in f:
                file_data.append(get_data(line))
                num_lines += 1

        if num_lines > max_num_lines:
            max_num_lines = num_lines

        category_code = category_codes[get_category(fname)]
        categories = [category_code] * len(file_data)
        x_nested.append(file_data)
        y_nested.append(categories)

    # intersperse
    x_final = []
    y_final = []
    for j in range(0, max_num_lines):
        for i in range(0, num_files):
            try:
                x_final.append(x_nested[i][j])
                y_final.append(y_nested[i][j])
            except:
                continue

    return Bunch(
        DESCR="complete set",
        data=x_final,
        target=np.array(y_final),
        target_names=category_codes.keys(),
        filenames=fnames
    )

###############################################################################
# Load some categories from the training set
#categories = [
#    'alt.atheism',
#    'talk.religion.misc',
#]
# Uncomment the following to do the analysis on all the categories
#categories = None

#print("Loading 20 newsgroups dataset for categories:")
#print(categories)

#data = fetch_20newsgroups(subset='train', categories=categories)

data = get_data_frame(
    opts.data_dir,
    lambda line: json.loads(line)['content'])

categories = data.target_names

print("%d documents" % len(data.filenames))
print("%d categories" % len(data.target_names))
print()

###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

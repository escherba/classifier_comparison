# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

from __future__ import print_function

from pprint import pprint
from time import time
import logging
import sys
import json

from optparse import OptionParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from lfcorpus_utils import get_data_frame

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
    ('vect', CountVectorizer(lowercase=True, stop_words="english", max_df=0.30)),
    ('tfidf', TfidfTransformer(sublinear_tf=True)),
    ('norm', Normalizer()),
    ('clf', SGDClassifier(loss='hinge', penalty='elasticnet', n_iter=50)),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    #'vect__max_df': (0.20, 0.30, 0.40),
    #'vect__lowercase': (True, False),
    #'vect__stop_words': ('english', None),
    #'vect__max_features': (None, 5000, 10000, 50000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__sublinear_tf': (True, False),
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
    'clf__l1_ratio': (0.0, 0.1, 0.2),
    'clf__loss': ('hinge', 'log', 'modified_huber')
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

from __future__ import print_function

import json
import logging
import numpy as np
import pylab as pl

from argparse import ArgumentParser
from time import time
from utils.lfcorpus import get_data_frame

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = ArgumentParser()
op.add_argument("--vectorizer", type=str, default="tfidf",
                choices=["tfidf", "hashing"],
                help="Vectorizer to use")
op.add_argument("--n_features",
                action="store", type=int, default=2 ** 16,
                help="n_features when using the hashing vectorizer.")
op.add_argument("--method", default="SVD", type=str, choices=["NMF", "SVD"],
                help="Decomposition method to use")
op.add_argument("--data_dir", type=str,
                help="Data directory", required=True)

args = op.parse_args()
if args.method == "NMF":
    from sklearn.decomposition import NMF as Decomposition
elif args.method == "SVD":
    from sklearn.decomposition import TruncatedSVD as Decomposition
else:
    op.error("Invalid decomposition method")


data_train = get_data_frame(
    args.data_dir,
    lambda line: json.loads(line)['content'])

# split a training set and a test set
y_train = data_train.target

print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()
if args.vectorizer == "hashing":
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=args.n_features)
    X_train = vectorizer.transform(data_train.data)
    # mapping from integer feature name to original token string
    feature_names = None
elif args.vectorizer == "tfidf":
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.4,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
    duration = time() - t0
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()
    # mapping from integer feature name to original token string
    feature_names = np.asarray(vectorizer.get_feature_names())


# Perform PCA
pca = Decomposition(n_components=2)
ox = pca.fit_transform(X_train, y_train)

# Plot results split by classes
data_length = y_train.shape[0]
neg_y_train = np.array([1 ** data_length]) - y_train

component0 = ox[:, 0]
component1 = ox[:, 1]

positives0 = component0[y_train.nonzero()]
positives1 = component1[y_train.nonzero()]
negatives0 = component0[neg_y_train.nonzero()]
negatives1 = component1[neg_y_train.nonzero()]

pl.plot(positives0, positives1, 'r+')
pl.plot(negatives0, negatives1, 'b+')

categories = list(data_train['target_names'])
categories.reverse()
pl.legend(categories)
pl.show()

from __future__ import print_function

import json
import logging
import numpy as np
import pylab as pl
from argparse import ArgumentParser
from time import time

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

from utils.lfcorpus import get_data_frame
from lflearn.feature_extract import TextExtractor, FeaturePipeline


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


data = get_data_frame(
    args.data_dir,
    lambda line: json.loads(line))

# split a training set and a test set
y = data.target

print("Extracting features from the training dataset "
      "using a sparse vectorizer")
if args.vectorizer == "hashing":
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=args.n_features)
elif args.vectorizer == "tfidf":
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.4,
                                 stop_words='english')

content_pipeline = FeaturePipeline([
    ('cont1', TextExtractor('content')),
    ('vec', vectorizer),
])
preprocess = FeatureUnion([
    ('cp', content_pipeline),
    # feature pipeline of your choice
])

# Fit to data
t0 = time()
if args.vectorizer == "hashing":
    X = preprocess.transform(data.data)
elif args.vectorizer == "tfidf":
    X = preprocess.fit_transform(data.data)
duration = time() - t0
print("n_samples: %d, n_features: %d" % X.shape)
print()


# Perform PCA
pca = Decomposition(n_components=2)
ox = pca.fit_transform(X, y)

# Plot results split by classes
data_length = y.shape[0]
neg_y = np.array([1 ** data_length]) - y

component0 = ox[:, 0]
component1 = ox[:, 1]

positives0 = component0[y.nonzero()]
positives1 = component1[y.nonzero()]
negatives0 = component0[neg_y.nonzero()]
negatives1 = component1[neg_y.nonzero()]

pl.plot(positives0, positives1, 'r+')
pl.plot(negatives0, negatives1, 'b+')

categories = list(data['target_names'])
categories.reverse()
pl.legend(categories)
pl.show()

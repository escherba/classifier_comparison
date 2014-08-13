# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

from __future__ import print_function

from argparse import ArgumentParser

import json
from itertools import izip
from time import time
from sklearn.feature_extraction import text
from sklearn.cluster import AffinityPropagation as Decomposition
from utils.lfcorpus import get_data_frame

# parse commandline arguments
op = ArgumentParser()
op.add_argument("--n_samples", default=1000, type=int,
                help="number of samples to use")
op.add_argument("--n_features", default=1000, type=int,
                help="number of features to expect")
op.add_argument("--n_top_words", default=20, type=int,
                help="number of top words to print")
op.add_argument("--categories", nargs="+", type=str,
                help="Categories (e.g. spam, ham)")
op.add_argument("--data_dir", type=str, default=None,
                help="data directory", required=False)

args = op.parse_args()

# Load the 20 newsgroups dataset and vectorize it using the most common word
# frequency with TF-IDF weighting (without top 5% stop words)


print("Loading dataset and extracting TF-IDF features...")

if args.data_dir is None:
    # Load 20 newsgroups corpus
    print("loading 20 newsgroups corpus")
    from sklearn.datasets import fetch_20newsgroups
    if args.categories is None:
        categories = [
            'alt.atheism',
            'talk.religion.misc',
            'comp.graphics',
            'sci.space',
        ]
    else:
        categories = args.categories
    fields_to_remove = ()  # ('headers', 'footers', 'quotes')
    print("Loading 20 newsgroups dataset for categories:")
    print(categories if categories else "all")
    dataset = fetch_20newsgroups(subset='train', categories=categories,
                                 shuffle=True, random_state=42,
                                 remove=fields_to_remove)
else:
    if args.categories is not None:
        cat_filter = set(args.categories)
    else:
        cat_filter = None

    dataset = get_data_frame(
        args.data_dir,
        lambda line: json.loads(line)['content'],
        cat_filter=cat_filter)

vectorizer = text.CountVectorizer(max_df=0.95, max_features=args.n_features,
                                  lowercase=True, stop_words="english")


print("Vectorizing...")

t0 = time()
samples = dataset.data[:args.n_samples]
counts = vectorizer.fit_transform(samples)
tfidf = text.TfidfTransformer(norm="l2", use_idf=True).fit_transform(counts)
print("done in %0.3fs." % (time() - t0))

# Fit the model
print("Fitting the model on with n_samples=%d and n_features=%d..."
      % (args.n_samples, args.n_features))

t0 = time()
d = Decomposition()
nmf = d.fit(tfidf)
print("done in %0.3fs." % (time() - t0))


# Fit the model
print("Predicting labels...")

t0 = time()
labels = d.predict(tfidf)
print("done in %0.3fs." % (time() - t0))

for sample, label in izip(samples, labels):
    print(sample, label)

# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

from __future__ import print_function

from argparse import ArgumentParser

import json

from time import time
from sklearn.feature_extraction import text
from utils.lfcorpus import get_data_frame

# parse commandline arguments
op = ArgumentParser()
op.add_argument("--n_samples", default=1000, type=int,
                help="number of samples to use")
op.add_argument("--n_features", default=1000, type=int,
                help="number of features to expect")
op.add_argument("--n_topics", default=10, type=int,
                help="number of topics to find")
op.add_argument("--method", default="NMF", type=str, choices=["NMF", "SVD"],
                help="Decomposition method to use")
op.add_argument("--n_top_words", default=20, type=int,
                help="number of top words to print")
op.add_argument("--categories", nargs="+", type=str,
                help="number of top words to print")
op.add_argument("--data_dir", type=str,
                help="data directory", required=True)

args = op.parse_args()
if args.method == "NMF":
    from sklearn.decomposition import NMF as Decomposition
elif args.method == "SVD":
    from sklearn.decomposition import TruncatedSVD as Decomposition
else:
    op.error("Invalid decomposition method")

# Load the 20 newsgroups dataset and vectorize it using the most common word
# frequency with TF-IDF weighting (without top 5% stop words)

t0 = time()
print("Loading dataset and extracting TF-IDF features...")

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
counts = vectorizer.fit_transform(dataset.data[:args.n_samples])
tfidf = text.TfidfTransformer(norm="l2", use_idf=True).fit_transform(counts)
print("done in %0.3fs." % (time() - t0))

# Fit the model
print("Fitting the %s model on with n_samples=%d and n_features=%d..."
      % (args.method, args.n_samples, args.n_features))
nmf = Decomposition(n_components=args.n_topics).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

# Inverse the vectorizer vocabulary to be able
feature_names = vectorizer.get_feature_names()

for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-args.n_top_words - 1:-1]]))
    print()

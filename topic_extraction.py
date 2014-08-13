# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

from __future__ import print_function

from argparse import ArgumentParser

import colorama
from colorama import init as colorama_init
colorama_init()

import json

from itertools import izip
from time import time
from scipy import sparse
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
                help="Categories (e.g. spam, ham)")
op.add_argument("--data_dir", type=str, default=None,
                help="data directory", required=False)

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

samples = dataset.data[:args.n_samples]
counts = vectorizer.fit_transform(samples)
tfidf = text.TfidfTransformer(norm="l2", use_idf=True).fit_transform(counts)
print("done in %0.3fs." % (time() - t0))

# Fit the model
print("Fitting the %s model on with n_samples=%d and n_features=%d..."
      % (args.method, args.n_samples, args.n_features))

num_topics = args.n_topics
nmf = Decomposition(n_components=num_topics).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

sparse_nmf = sparse.csr_matrix(nmf.components_)
topics_x_comments = tfidf.dot(sparse_nmf.transpose())

feature_names = vectorizer.get_feature_names()

topic_names = []
for j, topic in enumerate(nmf.components_):
    print("Topic #%d:" % j)
    weight_words = [(topic[i], feature_names[i])
                    for i in topic.argsort()[:-args.n_top_words - 1:-1]]
    print(weight_words)
    top_word = weight_words[0][1]
    topic_names.append("%d-%s" % (j, top_word))
    print()

for sample, topics in izip(samples, topics_x_comments):
    m = topics.todense()
    found_topics = sorted([(round(m[0, i], 3), topic_names[i])
                           for i in range(0, num_topics)], reverse=True)
    print(colorama.Fore.RESET + sample)
    print(colorama.Fore.WHITE + ', '.join(ft[1] for ft in found_topics[:3]))
    print(colorama.Fore.GREEN + str(found_topics[:3]))
    print()

from __future__ import print_function
from argparse import ArgumentParser

import colorama
from colorama import init as colorama_init
colorama_init()

import json

from itertools import izip, imap
from time import time
from scipy import sparse
#from sklearn.feature_extraction import text
from utils.lfcorpus import get_data_frame
from lsh_hdc.stats import safe_div, uncertainty_score
from pymaptools.iter import take
from functools import partial
from utils.feature_extract import FeaturePipeline, TextExtractor, \
    ChiSqBigramFinder
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score as NMI_score


def has_tag(tag, json_obj):
    imp_section = json_obj.get('impermium', {}) or {}
    imp_result = imp_section.get('result', {}) or {}
    imp_tags = imp_result.get('tag_details', {}) or {}
    return tag in imp_tags


is_bulk = partial(has_tag, 'bulk')
is_spam = partial(has_tag, 'spam')


class USumm(object):
    def __init__(self):
        self.categories = []
        self.topics = []

    def add(self, obj, topic):
        self.categories.append(is_spam(obj))
        self.topics.append(topic)

    def summarize(self):
        return dict(nmi=NMI_score(self.categories, self.topics),
                    u1=uncertainty_score(self.categories, self.topics),
                    u2=uncertainty_score(self.topics, self.categories))


# parse commandline arguments
op = ArgumentParser()
op.add_argument("--n_samples", default=10000, type=int,
                help="number of samples to use")
op.add_argument("--n_features", default=400, type=int,
                help="number of features to expect")
op.add_argument("--n_topics", default=10, type=int,
                help="number of topics to find")
op.add_argument("--show_topics", type=int, default=10,
                help="whether to list topic names")
op.add_argument("--show_comments", type=int, default=20,
                help="whether to list comments")
op.add_argument("--method", default="NMF", type=str, choices=["NMF", "SVD"],
                help="Decomposition method to use")
op.add_argument("--n_top_words", default=20, type=int,
                help="number of top words to print")
op.add_argument("--categories", nargs="+", type=str,
                help="Categories (e.g. spam, ham)")
op.add_argument("--topic_ratio", default=8.0, type=float,
                help="Ratio by which fisrt topic must be heavier than second")
op.add_argument("--word_ratio", default=1.33, type=float,
                help="Ratio by which fisrt word must be heavier than second"
                     "(for labeling topics)")
op.add_argument("--data_dir", type=str, default=None,
                help="data directory", required=False)
op.add_argument("--input", type=str, default=None,
                help="input file", required=False)

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


def show_mac(obj):
    return obj['object']['content']


if args.data_dir is None and args.input is None:
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
    content_column = None
    dataset = fetch_20newsgroups(subset='train', categories=categories,
                                 shuffle=True, random_state=42,
                                 remove=fields_to_remove)
    data = dataset.data[:args.n_samples]
    samples = data
    show_sample = str
    Y = None
elif args.data_dir is not None and args.input is None:
    if args.categories is not None:
        cat_filter = set(args.categories)
    else:
        cat_filter = None

    content_column = 'content'
    dataset = get_data_frame(
        args.data_dir,
        lambda line: json.loads(line)[content_column],
        cat_filter=cat_filter)
    data = dataset.data[:args.n_samples]
    samples = data
    show_sample = str
    Y = None
else:
    with open(args.input, 'r') as fh:
        dataset = imap(json.loads, fh)
        data = take(args.n_samples, dataset)
    content_column = 'content'
    samples = [s['object'] for s in data]
    show_sample = show_mac
    Y = [is_spam(s) for s in data]

content_pipeline = FeaturePipeline([
    ('cont1', TextExtractor(content_column)),
    ('vectf', TfidfVectorizer(sublinear_tf=True, max_df=0.3, lowercase=True,
                              max_features=args.n_features))
])
colloc_pipeline = FeaturePipeline([
    ('cont1', TextExtractor(content_column)),
    ('coll', ChiSqBigramFinder(score_thr=50)),
    ('vectc', DictVectorizer()),
])
preprocess = FeatureUnion([
    ('w', content_pipeline),
    ('bi', colloc_pipeline)
])


#vectorizer = text.CountVectorizer(max_df=0.95, max_features=args.n_features,
#                                  lowercase=True, stop_words="english")

tfidf = preprocess.fit_transform(samples)
#tfidf = text.TfidfTransformer(norm="l2", use_idf=True).fit_transform(counts)
print("done in %0.3fs." % (time() - t0))

# Fit the model
print("Fitting the %s model on with n_samples=%d and n_features=%d..."
      % (args.method, args.n_samples, args.n_features))

nmf = Decomposition(n_components=args.n_topics).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

sparse_nmf = sparse.csr_matrix(nmf.components_)
topics_x_comments = tfidf.dot(sparse_nmf.transpose())

feature_names = preprocess.get_feature_names()

topic_names = []
for j, topic in enumerate(nmf.components_):
    weight_words = [(topic[i], feature_names[i])
                    for i in topic.argsort()[:-args.n_top_words - 1:-1]]
    word1, word2 = weight_words[:2]
    prefix = str(j) + '_'
    topic_name = prefix + word1[1] \
        if safe_div(word1[0], word2[0]) >= args.word_ratio \
        else prefix + word1[1] + '-' + word2[1]
    topic_names.append(topic_name)


if args.show_topics is not None:
    print()
    print("Topics (showing %d)........" % args.show_topics)
    print()
    n = min(args.show_topics, len(topic_names))
    for i, topic_name in enumerate(take(n, topic_names)):
        print(i + 1, topic_name)


if args.show_comments is not None:
    print()
    print("Comments (%d well-assigned)........" % args.show_comments)
    print()
    n = min(args.show_comments, len(data))
    comments_shown = 0
    us = USumm()
    for sample, topics in izip(data, topics_x_comments):
        if comments_shown > args.show_comments:
            break
        m = topics.todense()
        found_topics = sorted(((round(m[0, i], 3), name)
                              for i, name in enumerate(topic_names)),
                              reverse=True)
        topic1, topic2 = found_topics[:2]
        assigned_topic = topic1[1] \
            if safe_div(topic1[0], topic2[0]) >= args.topic_ratio \
            else None
        us.add(sample, '(none)')
        if assigned_topic is not None:
            us.add(sample, assigned_topic)
            print(colorama.Fore.WHITE + assigned_topic + ':',
                  colorama.Fore.RESET + show_sample(sample))
            print()
            comments_shown += 1
    print("NMI coeff: %s" % us.summarize())


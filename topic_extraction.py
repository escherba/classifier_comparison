from __future__ import print_function
from argparse import ArgumentParser

import json

from itertools import izip, imap
from time import time
from scipy import sparse
from utils.lfcorpus import get_data_frame
from lsh_hdc.stats import safe_div, uncertainty_score
from pymaptools.iter import take
from functools import partial
from collections import defaultdict, Counter
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


def get_id(json_obj):
    return json_obj['object']['post_id']


class USumm(object):
    def __init__(self):
        self.categories = []
        self.topics = []
        self.topic_map = defaultdict(Counter)
        self.default_pred = '(none)'

    def add(self, obj, label_true, label_pred):
        self.categories.append(label_true)
        self.topics.append(label_pred)
        self.topic_map[label_pred][label_true] += 1

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
op.add_argument("--show_topics", action="store_true",
                help="whether to list topic names")
op.add_argument("--show_comments", action="store_true",
                help="whether to list comments")
op.add_argument("--method", default="NMF", type=str, choices=["NMF", "SVD"],
                help="Decomposition method to use")
op.add_argument("--ground_tag", default="spam", type=str,
                choices=["spam", "bulk"], help="Tag to compare to")
op.add_argument("--n_top_words", default=20, type=int,
                help="number of top words to print")
op.add_argument("--categories", nargs="+", type=str,
                help="Categories (e.g. spam, ham)")
op.add_argument("--topic_ratio", default=4.0, type=float,
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


is_ground_true = partial(has_tag, args.ground_tag)

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
    # Y = None
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
    # Y = None
else:
    with open(args.input, 'r') as fh:
        dataset = imap(json.loads, fh)
        data = take(args.n_samples, dataset)
    content_column = 'content'
    samples = [s['object'] for s in data]
    show_sample = show_mac
    # Y = [is_ground_true(s) for s in data]

content_pipeline = FeaturePipeline([
    ('cont1', TextExtractor(content_column)),
    ('vectf', TfidfVectorizer(sublinear_tf=True, max_df=0.3, lowercase=True,
                              max_features=args.n_features, use_idf=True,
                              stop_words="english", norm='l1'))
])
colloc_pipeline = FeaturePipeline([
    ('cont1', TextExtractor(content_column)),
    ('coll', ChiSqBigramFinder(score_thr=50)),
    ('vectc', DictVectorizer()),
])
preprocess = FeatureUnion([
    ('w', content_pipeline),
    #('bi', colloc_pipeline)
])


tfidf = preprocess.fit_transform(samples)
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


us = USumm()

if args.show_comments is not None:
    print()
    print("Comments (well-assigned)........")
    print()
    for sample, topics in izip(data, topics_x_comments):
        m = topics.todense()
        found_topics = sorted(((round(m[0, i], 3), name)
                              for i, name in enumerate(topic_names)),
                              reverse=True)
        topic1, topic2 = found_topics[:2]
        assigned_topic = topic1[1] \
            if safe_div(topic1[0], topic2[0]) >= args.topic_ratio \
            else us.default_pred
        us.add(sample, is_ground_true(sample), assigned_topic)


table_format = "{: <20} {: <30}"
if args.show_topics is not None:
    print()
    print("Topics (showing %d)........" % args.n_topics)
    print()
    ground_map = Counter()
    for topic_name in topic_names:
        c = us.topic_map[topic_name]
        ground_map.update(c)
        print(table_format.format(topic_name, c.items()))

    c = us.topic_map[us.default_pred]
    ground_map.update(c)
    print(table_format.format(us.default_pred, c.items()))
    print(table_format.format("total", ground_map.items()))
    print()
    print("NMI coeff: %s" % us.summarize())
    print()

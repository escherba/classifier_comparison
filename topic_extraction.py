from __future__ import print_function
from argparse import ArgumentParser

import json
import logging

import sys
from math import ceil
from itertools import izip, imap
from time import time
from scipy import sparse
from utils.lfcorpus import get_data_frame
from lsh_hdc.stats import safe_div, ClusteringComparator
from pymaptools.iter import take
from functools import partial
from utils.feature_extract import FeaturePipeline, TextExtractor, \
    ChiSqBigramFinder
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# setup logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
LOG.addHandler(stream_handler)


def has_common_tags(tags, json_obj):
    imp_section = json_obj.get('impermium', {}) or {}
    imp_result = imp_section.get('result', {}) or {}
    imp_tags = set(imp_result.get('tag_details', {}).keys() or [])
    return len(tags & imp_tags) > 0


def get_id(json_obj):
    return json_obj['object']['post_id']


TAG_MAP = dict(
    profanity={'strong_profanity', 'mild_profanity'},
    insult={'strong_insult', 'mild_insult'},
    spam={'spam'},
    bulk={'bulk'}
)

# parse commandline arguments
op = ArgumentParser()
op.add_argument("--n_samples", default=20000, type=int,
                help="number of samples to use")
op.add_argument("--n_topics", default=40, type=int,
                help="number of topics to find")
op.add_argument("--features_per_topic", default=3.0, type=float,
                help="number of features to per topic")
op.add_argument("--show_topics", action="store_true",
                help="whether to list topic names")
op.add_argument("--method", default="NMF", type=str, choices=["NMF", "SVD"],
                help="Decomposition method to use")
op.add_argument("--ground_tag", default=None, type=str,
                choices=TAG_MAP.keys(), help="Tag set to compare against"
                "If None, will rely on user_id for MAC content")
op.add_argument("--n_top_words", default=20, type=int,
                help="number of top words to print")
op.add_argument("--categories", nargs="+", type=str,
                help="Categories (e.g. spam, ham)")
op.add_argument("--topic_ratio", default=4.0, type=float,
                help="Ratio by which fisrt topic must be heavier than second")
op.add_argument("--word_ratio", default=1.618, type=float,
                help="Ratio by which fisrt word should be heavier than second"
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
LOG.info("Loading dataset and extracting TF-IDF features...")


content_column = u'content'

if args.data_dir is None and args.input is None:
    # Load 20 newsgroups corpus
    LOG.info("loading 20 newsgroups corpus")
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
    LOG.info("Loading 20 newsgroups dataset for categories: %s"
             % (categories if categories else "all"))
    dataset = fetch_20newsgroups(subset='train', categories=categories,
                                 shuffle=True, random_state=42,
                                 remove=fields_to_remove)
    data = []
    for i in range(0, min(args.n_samples, len(dataset.data))):
        data.append({content_column: dataset.data[i],
                     'category': dataset.target[i]})
    get_ground_truth = lambda o: dataset.target_names[o['category']]
    samples = data
    # Y = None
elif args.data_dir is not None and args.input is None:
    get_ground_truth = partial(has_common_tags, TAG_MAP[args.ground_tag])
    if args.categories is not None:
        cat_filter = set(args.categories)
    else:
        cat_filter = None

    dataset = get_data_frame(
        args.data_dir,
        lambda line: json.loads(line),
        cat_filter=cat_filter)
    data = dataset.data[:args.n_samples]
    samples = data
    # Y = None
else:
    if args.ground_tag is None:
        get_ground_truth = lambda o: o['object']['user_id']
    else:
        get_ground_truth = partial(has_common_tags, TAG_MAP[args.ground_tag])
    with open(args.input, 'r') as fh:
        dataset = imap(json.loads, fh)
        data = take(args.n_samples, dataset)
    samples = [s['object'] for s in data]
    # Y = [get_ground_truth(s) for s in data]

n_features = int(ceil(args.n_topics * args.features_per_topic))

content_pipeline = FeaturePipeline([
    ('cont1', TextExtractor(content_column)),
    ('vectf', TfidfVectorizer(sublinear_tf=True, max_df=0.3, lowercase=True,
                              max_features=n_features, use_idf=True,
                              stop_words="english", norm='l1'))
])
colloc_pipeline = FeaturePipeline([
    ('cont1', TextExtractor(content_column)),
    ('coll', ChiSqBigramFinder(score_thr=50)),
    ('vectc', DictVectorizer()),
])
preprocess = FeatureUnion([
    ('w', content_pipeline),
    # ('bi', colloc_pipeline)
])


tfidf = preprocess.fit_transform(samples)
LOG.info("done in %0.3fs." % (time() - t0))

# Fit the model
LOG.info("Fitting the %s model with n_samples=%d, n_features=%d, "
         "n_topics=%d..." %
         (args.method, args.n_samples, n_features, args.n_topics))

nmf = Decomposition(n_components=args.n_topics).fit(tfidf)
LOG.info("done in %0.3fs." % (time() - t0))

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


cs = ClusteringComparator({'ratio': args.topic_ratio})

for sample, topics in izip(data, topics_x_comments):
    m = topics.todense()
    found_topics = sorted(((round(m[0, i], 3), name)
                           for i, name in enumerate(topic_names)),
                          reverse=True)
    topic1, topic2 = found_topics[:2]
    assigned_topic = topic1[1] \
        if safe_div(topic1[0], topic2[0]) >= args.topic_ratio \
        else cs.default_pred
    cs.add(get_ground_truth(sample), assigned_topic)

if args.show_topics:
    table_format = u"{: <20} {: <30}"
    print()
    for topic in topic_names:
        print(table_format.format(topic,
                                  cs.summarize_pred(topic,
                                                    formatted=True)))
    print(table_format.format(cs.default_pred,
                              cs.summarize_pred(cs.default_pred,
                                                formatted=True)))
    print(table_format.format("total", cs.true_counts(formatted=True)))

    print()

print(json.dumps(cs.summarize()))

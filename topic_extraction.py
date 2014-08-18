from argparse import ArgumentParser

import json
import logging

import sys
from itertools import izip, imap
from time import time
from utils.lfcorpus import get_data_frame
from lsh_hdc.stats import safe_div, ClusteringComparator
from pymaptools.iter import take
from functools import partial
from utils.feature_extract import FeaturePipeline, TextExtractor
import numpy as np
from scipy.sparse.csr import csr_matrix
from sklearn.pipeline import FeatureUnion
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


def get_imp(json_obj):
    imp_section = json_obj.get('impermium', {}) or {}
    imp_result = imp_section.get('result', {}) or {}
    return imp_result


def has_common_tags(tags, json_obj):
    imp_result = get_imp(json_obj)
    imp_tags = set(imp_result.get('tag_details', {}).keys() or [])
    return len(tags & imp_tags) > 0


def get_imp_lang(json_obj):
    imp_result = get_imp(json_obj)
    return imp_result.get('language')


def get_langid_lang(json_obj):
    langid_section = json_obj.get('langid', {}) or {}
    langid_result = langid_section.get('result', {}) or {}
    return langid_result.get('lang')


def get_id(json_obj):
    return json_obj['object']['post_id']


def get_user(json_obj):
    return json_obj['object']['user_id']


TAG_MAP = dict(
    profanity={'strong_profanity', 'mild_profanity'},
    insult={'strong_insult', 'mild_insult'},
    spam={'spam'},
    bulk={'bulk'}
)

ATTR_MAP = dict(
    user=get_user,
    lang=get_langid_lang,
    lang_imp=get_imp_lang
)

# parse commandline arguments
op = ArgumentParser()
op.add_argument("--n_samples", default=None, type=int,
                help="number of samples to use")
op.add_argument("--n_topics", default=None, type=int,
                help="number of topics to find")
op.add_argument("--n_features", default=None, type=int,
                help="number of topics to find")
op.add_argument("--features_per_topic", default=6.0, type=float,
                help="number of features per topic")
op.add_argument("--show_topics", action="store_true",
                help="whether to list topic names")
op.add_argument("--show_top_words", type=int, required=False,
                help="how many top words to list for each topic")
op.add_argument("--method", default="NMF", type=str, choices=["NMF", "SVD"],
                help="Decomposition method to use")
op.add_argument("--nmf_max_iter", default=400, type=int, required=False,
                help="maximum number of iterations for NMF")
op.add_argument("--nmf_beta", default=1.0, type=float, required=False,
                help="Degree of sparseness, if sparseness is not None. "
                "Larger values mean more sparseness.")
op.add_argument("--nmf_eta", default=0.1, type=float, required=False,
                help="Degree of correctness to maintain, if sparsity set. "
                "Smaller values mean larger error.")
op.add_argument("--nmf_sparseness", default=None, type=str, required=False,
                choices=["data", "components"],
                help="Where to enforce sparsity in the model")
op.add_argument("--nmf_nls_max_iter", default=4000, type=int, required=False,
                help="maximum number of iterations for NMF NLS subproblem")
op.add_argument("--nmf_init", default="nndsvd", type=str,
                choices=["nndsvd", "nndsvda", "nndsvdar", "random"],
                help="Initialization method for NMF")
op.add_argument("--H_matrix", default="multiply", type=str, required=False,
                choices=["standard", "multiply"], help="H_matrix algorithm")
op.add_argument("--ground_tag", default=None, type=str,
                choices=TAG_MAP.keys(), help="Tag set to compare against")
op.add_argument("--ground_attr", default="user", type=str,
                choices=ATTR_MAP.keys(), help="Attribute to compare against")
op.add_argument("--categories", nargs="+", type=str, default=None,
                help="Categories (e.g. spam, ham)")
op.add_argument("--topic_ratio", default=[2.2], type=float, nargs='+',
                help="Ratio(s) by which 1st topic must be heavier than 2nd")
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

content_column = u'content'

n_samples = args.n_samples \
    if args.n_samples is not None \
    else float('inf')

n_topics = args.n_topics
categories = args.categories

t0 = time()
LOG.info("Loading dataset...")

if args.data_dir is None and args.input is None:
    # Load 20 newsgroups corpus
    LOG.info("loading 20 newsgroups corpus")
    from sklearn.datasets import fetch_20newsgroups
    if categories is None:
        categories = [
            'alt.atheism',
            'talk.religion.misc',
            'comp.graphics',
            'sci.space',
        ]

    if n_topics is None:
        n_topics = len(categories)

    fields_to_remove = ()  # ('headers', 'footers', 'quotes')
    LOG.info("Loading 20 newsgroups dataset for categories: %s"
             % (categories if categories else "all"))
    dataset = fetch_20newsgroups(subset='train', categories=categories,
                                 shuffle=True, random_state=42,
                                 remove=fields_to_remove)
    data = []
    for i in range(0, min(n_samples, len(dataset.data))):
        data.append({content_column: dataset.data[i],
                     'category': dataset.target[i]})
    get_ground_truth = lambda o: dataset.target_names[o['category']]
    samples = data

elif args.data_dir is not None and args.input is None:
    get_ground_truth = partial(has_common_tags, TAG_MAP[args.ground_tag])
    if categories is not None:
        cat_filter = set(categories)
        if n_topics is None:
            n_topics = len(categories)
    else:
        cat_filter = None

    dataset = get_data_frame(
        args.data_dir,
        lambda line: json.loads(line),
        cat_filter=cat_filter)
    data = dataset.data[:n_samples]

    samples = data

elif args.input is not None:
    if args.ground_tag is not None:
        get_ground_truth = partial(has_common_tags, TAG_MAP[args.ground_tag])
    elif args.ground_attr is not None:
        get_ground_truth = ATTR_MAP[args.ground_attr]
    else:
        raise ValueError("neither ground_tag nor ground_attr specified")

    with open(args.input, 'r') as fh:
        dataset = imap(json.loads, fh)
        data = take(n_samples, dataset)
    samples = [s['object'] for s in data]

else:
    raise ValueError("No input sources specified.")


if n_samples == float('inf'):
    n_samples = len(data)
    assert n_samples >= 2

if n_topics is None:
    n_topics = 2

if args.n_features is None:
    n_features = n_topics + 1
else:
    n_features = args.n_features


content_pipeline = FeaturePipeline([
    ('cont1', TextExtractor(content_column)),
    ('vectf', TfidfVectorizer(sublinear_tf=True, max_df=0.3, lowercase=True,
                              max_features=n_features, use_idf=True,
                              stop_words="english", norm='l1'))
])
preprocess = FeatureUnion([
    ('w', content_pipeline),
])

LOG.info("done in %0.3fs." % (time() - t0))

LOG.info("Extracting TF-IDF features...")
tfidf = preprocess.fit_transform(samples)
LOG.info("done in %0.3fs." % (time() - t0))

opts = dict(n_components=n_topics)
if args.method == "NMF":
    opts['init'] = args.nmf_init
    opts['max_iter'] = args.nmf_max_iter
    opts['nls_max_iter'] = args.nmf_nls_max_iter
    opts['sparseness'] = args.nmf_sparseness
    opts['beta'] = args.nmf_beta
    opts['eta'] = args.nmf_eta

# Fit the model
LOG.info("Fitting {} model with n_samples={}, n_features={}, opts={}..."
         .format(args.method, n_samples, n_features, opts))

nmf_model = Decomposition(**opts)

if args.H_matrix == 'multiply':
    LOG.info("Obtaining H matrix by multiplying W by tfidf matrix ({})"
             .format(args.H_matrix))
    nmf_model.fit(tfidf)
    W_matrix = nmf_model.components_
    sparse_H = tfidf.dot(csr_matrix(W_matrix).transpose())
    H_matrix = np.asarray(sparse_H.todense())
else:
    LOG.info("Using provided H matrix ({})".format(args.H_matrix))
    H_matrix = nmf_model.fit_transform(tfidf)
    W_matrix = nmf_model.components_

reconstruction_err = nmf_model.reconstruction_err_

LOG.info("done in %0.3fs." % (time() - t0))

LOG.info("Labeling topics...")
feature_names = preprocess.get_feature_names()

topic_names = []
max_words = max(2, args.show_top_words) \
    if args.show_top_words is not None \
    else 2

for j, topic in enumerate(W_matrix):
    weight_words = [(topic[i], feature_names[i])
                    for i in topic.argsort()[:-max_words - 1:-1]]
    word1, word2 = weight_words[:2]
    prefix = str(j) + '_'
    topic_name = prefix + word1[1] \
        if safe_div(abs(word1[0]), abs(word2[0])) >= args.word_ratio \
        else prefix + word1[1] + '-' + word2[1]
    topic_names.append(topic_name)
    if args.show_top_words is not None:
        print topic_name, u' '. join(ww[1] for ww in weight_words)


LOG.info("done in %0.3fs." % (time() - t0))

LOG.info("Aggregating stats...")
for topic_ratio in args.topic_ratio:
    LOG.info("Aggregating stats for topic ratio {}".format(topic_ratio))
    cs = ClusteringComparator({'ratio': topic_ratio,
                               'reconstruction_err': reconstruction_err})

    # assign topics to comments
    for sample, topics in izip(data, H_matrix):
        top_topics = [(topics[i], topic_names[i])
                      for i in topics.argsort()[:-3:-1]]
        if len(top_topics) > 1:
            topic1, topic2 = top_topics
            assigned_topic = topic1[1] \
                if safe_div(abs(topic1[0]), abs(topic2[0])) >= topic_ratio \
                else cs.default_pred
        else:
            assigned_topic = top_topics[0][1]
        cs.add(get_ground_truth(sample), assigned_topic)

    if args.show_topics:
        print cs.cross_tab(row_order=-1, pct=True)

    unclust_key = '_unclust'
    summary = cs.summarize()
    assert unclust_key not in summary
    summary[unclust_key] = cs.freq_ratio_pred(cs.default_pred)
    print json.dumps(summary)

    if args.show_topics:
        print

LOG.info("done in %0.3fs." % (time() - t0))

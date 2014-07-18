# Adapted by:       Eugene Scherba <escherba@gmail.com>
# Original Authors: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#                   Olivier Grisel <olivier.grisel@ensta.org>
#                   Mathieu Blondel <mathieu@mblondel.org>
#                   Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

import logging
import json
import sys
import os
import re

from time import time
from argparse import ArgumentParser

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, \
    HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier, SGDClassifier, \
    LogisticRegression, Perceptron, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
# from sklearn import svm
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, BaseDiscreteNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.utils.extmath import density
from sklearn.metrics import roc_curve, auc, f1_score, classification_report, \
    confusion_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import StandardScaler, Normalizer

from utils import to_csv, split_num
from utils.lfcorpus import get_data_frame, get_data_frames
from utils.feature_extract import with_l1_feature_selection, TextExtractor, \
    FeaturePipeline, PCAPipeline, ChiSqBigramFinder

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# parse commandline arguments
op = ArgumentParser()
op.add_argument("--report",
                action="store_true", dest="print_report",
                help="Print a detailed classification report.")
op.add_argument("--chi2_select",
                action="store", type=int, dest="select_chi2",
                help="Select some number of features using a chi-squared test")
op.add_argument("--confusion_matrix",
                action="store_true", dest="print_cm",
                help="Print the confusion matrix.")
op.add_argument("--top_terms", type=int, dest="top_terms",
                help="Print most discriminative terms per class.")
op.add_argument("--vectorizer", type=str, default="tfidf",
                choices=["tfidf", "hashing"],
                help="Vectorizer to use")
op.add_argument("--n_features",
                action="store", type=int, default=2 ** 16,
                help="n_features when using the hashing vectorizer.")
op.add_argument("--data_dir", type=str,
                help="data directory")
op.add_argument("--data_train", type=str,
                help="data directory")
op.add_argument("--data_test", type=str,
                help="data directory")
op.add_argument("--output_dir", type=str, help="output path (for features)")
op.add_argument("--output", type=str, help="output path")
op.add_argument("--output_roc", type=str,  help="output path (ROC)")

opts = op.parse_args()

content_column = None
if opts.data_train and opts.data_test:
    print("manually specified corpus")
    content_column = "content"
    data_train = get_data_frame(
        opts.data_train,
        lambda line: json.loads(line),
        extension=".timetest")
    data_test = get_data_frame(
        opts.data_test,
        lambda line: json.loads(line),
        extension=".timetest")

elif opts.data_dir is None:
    # Load 20 newsgroups corpus
    print("loading 20 newsgroups corpus")
    from sklearn.datasets import fetch_20newsgroups
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    fields_to_remove = ()  # ('headers', 'footers', 'quotes')
    print("Loading 20 newsgroups dataset for categories:")
    print(categories if categories else "all")
    data_train = fetch_20newsgroups(subset='train', categories=categories,
                                    shuffle=True, random_state=42,
                                    remove=fields_to_remove)
    data_test = fetch_20newsgroups(subset='test', categories=categories,
                                   shuffle=True, random_state=42,
                                   remove=fields_to_remove)
else:
    # Load custom corpus
    content_column = "content"
    print("loading custom corpus")
    data_train, data_test = get_data_frames(
        opts.data_dir,
        lambda line: json.loads(line))

categories = data_train.target_names

if len(data_train.data) == 0:
    logger.error("No training data loaded")
    sys.exit(1)

if len(data_test.data) == 0:
    logger.error("No testing data loaded")
    sys.exit(1)

print('data loaded')

y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()

PCA_components = 5


if opts.vectorizer == "hashing":
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=opts.n_features)
elif opts.vectorizer == "tfidf":
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.3,
                                 stop_words='english')


content_pipeline = FeaturePipeline([
    ('cont1', TextExtractor(content_column)),
    ('vec', vectorizer),
])
pca_pipeline = PCAPipeline([
    ('cont2', TextExtractor(content_column)),
    ('vectf', TfidfVectorizer(sublinear_tf=True, max_df=0.4,
                              stop_words='english')),
    ('pca', TruncatedSVD(n_components=PCA_components))
])
colloc_pipeline = FeaturePipeline([
    ('cont1', TextExtractor(content_column)),
    ('coll', ChiSqBigramFinder(score_thr=50)),
    ('vectc', DictVectorizer())
])
#lang_pipeline = FeaturePipeline([
#    ('cont3', TextExtractor(content_column)),
#    ('lang', FeatureLang()),
#    ('dvec', DictVectorizer()),
#])
#len_pipeline = FeaturePipeline([
#    ('cont4', TextExtractor(content_column)),
#    ('len', LengthVectorizer())
#])
preprocess = FeatureUnion([
    ('w', content_pipeline),
    ('bi', colloc_pipeline),
    #('lp', lang_pipeline),
    # ('mp', len_pipeline)
])

if opts.vectorizer == "hashing":
    X_train = preprocess.transform(data_train.data)
elif opts.vectorizer == "tfidf":
    X_train = preprocess.fit_transform(data_train.data)


duration = time() - t0
print("X_train: n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test dataset using the same vectorizer")
t0 = time()
X_test = preprocess.transform(data_test.data)
duration = time() - t0
print("X_test: n_samples: %d, n_features: %d" % X_test.shape)
print()


# mapping from integer feature name to original token string
if opts.vectorizer == "hashing":
    feature_names = None
elif opts.vectorizer == "tfidf":
    feature_names = np.asarray(preprocess.get_feature_names())
    assert feature_names.shape[0] == X_train.shape[1] == X_test.shape[1], \
        ("feature_names-len: %d, X-train-len:%d, X-test-len: %d" %
         (feature_names.shape[0], X_train.shape[1], X_test.shape[1]))


if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    print("done in %fs" % (time() - t0))
    print()
    if feature_names:
        feature_names = ch2.transform(feature_names)[0]


###############################################################################
# Benchmark classifiers

all_roc_data = []


def benchmark(clf, clf_descr=None):

    def print_top_terms(clf_descr, category, cpairs):
        to_csv(os.path.join(opts.output_dir,
                            "feat_%s_%s.csv" %
                            (clf_descr, category)), [(re.sub('^[A-Za-z]+__', '', t).encode("utf-8"), c) for t, c in cpairs])
        #print("%s\n%s" %
        #      (category, ' '.join((u"%s:%f" % (t, coeff))
        #                          for t, coeff in cpairs)))

    print('_' * 80)

    if clf_descr is None:
        clf_descr = str(clf).split('(')[0]

    print("Training: " + clf_descr)
    print(clf)
    t0 = time()
    try:
        clf.fit(X_train, y_train)
    except ValueError as e:
        logger.error(e)
        return (clf_descr, 0, 0, 0)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    try:
        pred = clf.predict(X_test)
    except TypeError as e:
        logger.error(e)
        return (clf_descr, 0, 0, 0)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    # Get F1 score
    score = f1_score(y_test, pred)
    print("f1-score:   %0.3f" % score)

    # Get ROC curve
    predict_proba = getattr(clf, "predict_proba", None)
    decision_function = getattr(clf, "decision_function", None)
    e = None
    probas_ = None
    try:
        if callable(predict_proba):
            probas_ = predict_proba(X_test)[:, 1]
        elif callable(decision_function):
            probas_ = decision_function(X_test)  # LinearSVC
    except ValueError as e:
        logger.error(e)
    if probas_ is not None and e is None:
        if opts.data_dir is None:
            if len(probas_.shape) > 1:
                roc_predicted = probas_[:, 3]
            else:
                roc_predicted = probas_
            roc_observed = [1 if y == 3 else 0 for y in y_test]
        else:
            roc_predicted = probas_
            roc_observed = y_test

        fpr, tpr, thresholds = roc_curve(roc_observed, roc_predicted)
        roc_auc = auc(fpr, tpr)
        labels = [clf_descr] * len(fpr)
        all_roc_data.extend(zip(labels, fpr, tpr))
        print("ROC area = %0.3f" % roc_auc)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.top_terms is not None and feature_names is not None:
            print("top %d keywords per class:" % opts.top_terms)

            tfnames = clf.transformer_.transform(feature_names)[0] \
                if clf.__class__.__name__.startswith("FeatureSelect") \
                else feature_names
            num_cat = len(categories)
            buckets = split_num(len(tfnames), num_cat)
            if num_cat == 2:  # Binomial classification

                coefficients = clf.coef_[0]

                # Class 0
                indices = np.argsort(coefficients)
                size0 = min(opts.top_terms, buckets[0])
                top_indices_0 = indices[:size0]
                cpairs = zip(tfnames[top_indices_0],
                             coefficients[top_indices_0])
                if not isinstance(clf, BaseDiscreteNB):  # filter coefficients
                    cpairs = [(f, -c) for (f, c) in cpairs if c < 0.0]
                print_top_terms(clf_descr, categories[0], cpairs)

                # Class 1
                indices = np.argsort(coefficients)
                size1 = min(opts.top_terms, buckets[1])
                top_indices_1 = indices[-size1:][::-1]
                cpairs = zip(tfnames[top_indices_1],
                             coefficients[top_indices_1])
                if not isinstance(clf, BaseDiscreteNB):  # filter coefficients
                    cpairs = [(f, c) for (f, c) in cpairs if c > 0.0]
                print_top_terms(clf_descr, categories[1], cpairs)

            else:                     # Multinomial classification
                for i, category in enumerate(categories):
                    coefficients = clf.coef_[i]
                    indices = np.argsort(coefficients)
                    size = min(opts.top_terms, buckets[i])
                    top_indices = indices[-size:][::-1]
                    cpairs = zip(tfnames[top_indices],
                                 coefficients[top_indices])
                    if not isinstance(clf, BaseDiscreteNB):  # filter coeffs
                        cpairs = [(f, c) for (f, c) in cpairs if c > 0.0]
                    print_top_terms(clf_descr, category, cpairs)

        print()

    if opts.print_report:
        print("classification report:")
        print(classification_report(y_test, pred, target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(confusion_matrix(y_test, pred))

    return [clf_descr, score, train_time, test_time]


results = [["Classifier", "Score", "Train.Time", "Test.Time"]]

for clf in (
    RidgeClassifier(alpha=8.0, solver="sparse_cg"),
    Perceptron(n_iter=50, alpha=1.0),
    PassiveAggressiveClassifier(n_iter=10, C=0.1),
    NearestCentroid(metric='cosine'),
    KNeighborsClassifier(metric='cosine', algorithm='brute', n_neighbors=6),
    MultinomialNB(alpha=1.5),
    BernoulliNB(alpha=0.2, binarize=None)
):
    print('=' * 80)
    print("Classifier: " + clf.__class__.__name__)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("LinearSVC with %s penalty" % penalty.upper())
    results.append(benchmark(
        LinearSVC(loss='l2', penalty=penalty, dual=False, tol=1e-3),
        "LinearSVC_" + penalty.upper()))


print('=' * 80)
print("LinearSVC with L1-based feature selection")
results.append(benchmark(
    with_l1_feature_selection(
        LinearSVC, C=0.1, dual=False, tol=1e-3
    )(),
    "LinearSVC_L1_featsel"))


for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("Logistic Regression with %s penalty" % penalty.upper())
    results.append(benchmark(
        LogisticRegression(penalty=penalty, dual=False, tol=1e-3),
        "LogisticRegression_" + penalty.upper()))


print('=' * 80)
print("LogisticRegression with L1-based feature selection")
results.append(benchmark(
    with_l1_feature_selection(
        LogisticRegression, C=0.42, dual=False, tol=1e-3
    )(),
    "LogisticRegression_L1_featsel"))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("SGD with %s penalty" % penalty.upper())
    results.append(benchmark(
        SGDClassifier(loss='log', alpha=1e-4, n_iter=50, penalty=penalty),
        "SGD_" + penalty.upper()))

# print('=' * 80)
# print("SGD with elasticnet penalty")
# results.append(benchmark(
#     SGDClassifier(loss='hinge', alpha=1e-4, n_iter=50, penalty='elasticnet',
#                   l1_ratio=0.10),
#     "SGD (elasticnet penalty)"))

print('=' * 80)
print("SGD L1 feature selection")
clf = with_l1_feature_selection(
    SGDClassifier, loss='log', alpha=0.00021, n_iter=10
    )(loss='log', alpha=.0001, n_iter=50)
results.append(benchmark(clf, "SGD_L1_featsel"))


# print('=' * 80)
# print("Radial kernal svc")
# results.append(benchmark(SVC(kernel='rbf')))


if opts.output_roc:
    print("Writing ROC curve data")
    to_csv(opts.output_roc, all_roc_data)

if opts.output:
    print("Writing scores data")
    to_csv(opts.output, results)

# Adapted by:       Eugene Scherba <escherba@gmail.com>
# Original Authors: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#                   Olivier Grisel <olivier.grisel@ensta.org>
#                   Mathieu Blondel <mathieu@mblondel.org>
#                   Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time

import csv
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics

from lfcorpus_utils import get_data_frames

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type=int, dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top_terms", type=int, dest="top_terms",
              help="Print most discriminative terms per class.")
op.add_option("--use_hashing", action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--data_dir", type=str,
              help="data directory")
op.add_option("--output", type=str,
              help="output path")


(opts, args) = op.parse_args()
if opts.data_dir is None:
    op.error('Data directory not given')
if opts.output is None:
    op.error('Output path not given')

if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


data_train, data_test = get_data_frames(
    opts.data_dir,
    lambda line: json.loads(line)['content'])

categories = data_train.target_names    # for case categories == None


# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test dataset using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    print("done in %fs" % (time() - t0))
    print()


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = np.asarray(vectorizer.get_feature_names())


###############################################################################
# Benchmark classifiers
def benchmark(clf, clf_descr=None):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(y_test, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.top_terms is not None and feature_names is not None:
            print("top %d keywords per class:" % opts.top_terms)
            for i, category in enumerate(categories[1:]):
                top_terms = np.argsort(clf.coef_[i])[-opts.top_terms:]
                print("%s\n %s" % (category, ' '.join(feature_names[top_terms])))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    if clf_descr is None:
        clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = [["Classifier", "Score", "Train.Time", "Test.Time"]]
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive")):

    # (KNeighborsClassifier(n_neighbors=10), "kNN")
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(
        LinearSVC(loss='l2', penalty=penalty, dual=False, tol=1e-3),
        "LinearSVC (" + penalty.upper() + " penalty)"))


class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(C=0.2,
                                      penalty="l1",
                                      dual=False,
                                      tol=1e-3)
        print("before feture selection: " + str(X.shape))
        X = self.transformer_.fit_transform(X, y)
        print("after feature selection: " + str(X.shape))
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

print('=' * 80)
print("LinearSVC with L1-based feature selection")
results.append(benchmark(L1LinearSVC(),
               "LinearSVC (L1 feature select)"))


for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(
        LogisticRegression(penalty=penalty, dual=False, tol=1e-3),
        "LogisticRegression (" + penalty.upper() + " penalty)"))


class L1Logistic(LogisticRegression):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LogisticRegression(C=0.2,
                                               penalty="l1",
                                               dual=False,
                                               tol=1e-3)
        print("before feture selection: " + str(X.shape))
        X = self.transformer_.fit_transform(X, y)
        print("after feature selection: " + str(X.shape))
        return LogisticRegression.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LogisticRegression.predict(self, X)


print('=' * 80)
print("LogisticRegression with L1-based feature selection")
results.append(benchmark(L1LinearSVC(),
               "LogisticRegression (L1 feature select)"))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    results.append(benchmark(
        SGDClassifier(loss='hinge', alpha=.0001, n_iter=50, penalty=penalty),
        "SGD (" + penalty.upper() + " penalty)"))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(
    SGDClassifier(loss='hinge', alpha=.0001, n_iter=50, penalty="elasticnet"),
    "SGD w/ elastic-net penalty"))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))


with open(opts.output, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=",",
                        quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    writer.writerows(results)
from __future__ import print_function

import logging
import csv
import json
import sys

from time import time
from argparse import ArgumentParser

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, \
    HashingVectorizer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier, SGDClassifier, \
    LogisticRegression, Perceptron, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
# from sklearn import svm
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import StandardScaler, Normalizer

from lfcorpus_utils import get_data_frames
from lfcorpus_utils import get_data_frame
from lf_feat_extract import with_l1_feature_selection, TextExtractor, \
    FeatureLang, LengthVectorizer, FeaturePipeline, PCAPipeline, \
    ChiSqBigramFinder



data_directory = "./corpora/dec19"
new_data_directory = "./corpora/dec20"

data_train = get_data_frame(
    data_directory,
    lambda line: json.loads(line),
    extension =".timetest")

new_data = get_data_frame(
    new_data_directory,
    lambda line: json.loads(line),
    extension =".timetest")

y_train = data_train.target
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.3,
                                 stop_words='english')

content_pipeline = FeaturePipeline([
    ('cont1', TextExtractor('content')),
    ('vec', vectorizer),
])
colloc_pipeline = FeaturePipeline([
    ('cont1', TextExtractor('content')),
    ('coll', ChiSqBigramFinder(score_thr=70)),
    ('vectc', FeatureHasher(input_type="string", non_negative=True))
])


preprocess = FeatureUnion([
    ('cp', content_pipeline),
    ('op', colloc_pipeline)
])
X_train = preprocess.fit_transform(data_train.data)
X_new = preprocess.transform(new_data.data)
model = LinearSVC(loss='l2',penalty='l2',tol=1e-3)
trained_model = model.fit(X_train,y_train)
stuff = trained_model.decision_function(X_new)

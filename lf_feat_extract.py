import logging
import numpy as np

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import base
from langid import classify as langid_classify

logger = logging.getLogger(__name__)


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def with_l1_feature_selection(class_T, **kwargs):
    """
    Add L1-feature selection (LASSO) to a classifier
    """
    class FeatureSelect(class_T):

        def fit(self, X, y):
            # The smaller C, the stronger the regularization.
            # The more regularization, the more sparsity.
            self.transformer_ = class_T(penalty="l1", **kwargs)
            logger.info("before feature selection: " + str(X.shape))
            X = self.transformer_.fit_transform(X, y)
            logger.info("after feature selection: " + str(X.shape))
            return class_T.fit(self, X, y)

        def predict(self, X):
            X = self.transformer_.transform(X)
            return class_T.predict(self, X)

    FeatureSelect.__name__ += '_' + class_T.__name__
    return FeatureSelect


class FeatureLang(base.BaseEstimator,
                  base.TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [{"lang": langid_classify(x)[0]} for x in X]

    def get_feature_names(self):
        return ["language"]


class TextExtractor(base.BaseEstimator,
                    base.TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        col = self.column
        return np.asarray([row[col] for row in X], dtype=np.unicode)

    def get_feature_names(self):
        return [self.column]


class LengthVectorizer(base.BaseEstimator,
                       base.TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [[len(x)] for x in X]

    def get_feature_names(self):
        return ["length"]

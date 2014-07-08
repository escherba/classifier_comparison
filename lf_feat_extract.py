import logging
import numpy as np

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from langid import classify as langid_classify
from sklearn import base, metrics
from sklearn.pipeline import Pipeline
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

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


class ChiSqBigramFinder(base.BaseEstimator,
                        base.TransformerMixin):

    def __init__(self, score_thr=50):
        self.score_thr = score_thr

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = []
        scorer = (BigramAssocMeasures.chi_sq, self.score_thr)
        for x in X:
            g_finder = BigramCollocationFinder.from_words(x)
            bigrams = g_finder.above_score(*scorer)
            processed = (repr(b) for b in bigrams)
            result.append(processed)
        return result


class TextExtractor(base.BaseEstimator,
                    base.TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        column = self.column
        return np.asarray([row[column] for row in X], dtype=np.unicode)

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


class PCAPipeline(Pipeline):
    def get_feature_names(self):
        component_count = self.steps[-1][1].n_components
        return ["pc" + str(x) for x in range(component_count)]


class FeaturePipeline(Pipeline):
    def get_feature_names(self):
        return self.steps[-1][1].get_feature_names()


class F1Scorer:
    def __init__(self):
        pass

    def __call__(self, estimator, X, y):
        pred = estimator.predict(X)
        f1_score = metrics.f1_score(y, pred)
        logger.info(estimator)
        logger.info(estimator.steps[-1])
        logger.info("F1-score: %f" % f1_score)
        logger.info("X.shape = " + str(len(X)))
        logger.info("y.shape = " + str(y.shape))
        return f1_score

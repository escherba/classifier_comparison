import logging
import re

from HTMLParser import HTMLParser
from nltk.stem import WordNetLemmatizer
from langid import classify as langid_classify
from sklearn import base, metrics
from sklearn.pipeline import Pipeline
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.regexp import WordPunctTokenizer

logger = logging.getLogger(__name__)


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def handle_entityref(self, name):
        # Ignore HTML entities (already unescaped)
        self.fed.append(u'&' + name)

    def get_data(self):
        return ''.join(self.fed)


def clean_html(html):
    """Remove HTML markup from the given string."""
    html = re.sub(r"(?s)<!--(.*?)-->[\n]?", "\\1", html)
    html = re.sub(r"<!--", "", html)
    if html == '':
        return ''
    s = MLStripper()
    s.feed(html)
    return s.get_data().strip()


class LemmaTokenizer(object):
    def __init__(self, tokenizer=None):
        self.wnl = WordNetLemmatizer()
        self.tokenizer = TreebankWordTokenizer \
            if tokenizer is None else tokenizer

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in self.tokenizer.tokenize(doc)]


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

    def __init__(self, tokenizer=None, score_thr=50):
        self.score_thr = score_thr
        self.tokenizer = WordPunctTokenizer() \
            if tokenizer is None else tokenizer

    def fit(self, X, y=None):
        return self

    @staticmethod
    def punct_filter_(w):
        """
        Match separating punctuation (commas, periods, but not colons, hyphens)
        """
        return w in {u'.', u',', u';', u'?', u'!', u'(', u')', u'[', u']'}

    def transform(self, X, y=None):
        result = []
        scorer = (BigramAssocMeasures.chi_sq, self.score_thr)
        for x in X:
            words = self.tokenizer.tokenize(x)
            g_finder = BigramCollocationFinder.from_words(words)
            g_finder.apply_word_filter(self.punct_filter_)
            bigrams = g_finder.above_score(*scorer)
            processed = {"%s:%s" % b: True for b in bigrams}
            result.append(processed)
        return result


class TextExtractor(base.BaseEstimator,
                    base.TransformerMixin):

    def __init__(self, column=None, lowercase=True):
        self.column = column
        self.lowercase = lowercase
        self.html_parser_ = HTMLParser()
        self.normalize_map = {k: None for k in (
            range(ord(u'\x00'), ord(u'\x08') + 1) +
            range(ord(u'\x0b'), ord(u'\x0c') + 1) +
            range(ord(u'\x0e'), ord(u'\x1f') + 1) +
            range(ord(u'\x7f'), ord(u'\x9f') + 1) +
            [ord(u'\uffff')] +
            [ord(u'\xad')] +
            range(ord(u'\u17b4'), ord(u'\u17b5') + 1) +
            range(ord(u'\u200b'), ord(u'\u200f') + 1) +
            range(ord(u'\u202a'), ord(u'\u202d') + 1) +
            range(ord(u'\u2060'), ord(u'\u2064') + 1) +
            range(ord(u'\u206a'), ord(u'\u206f') + 1) +
            [ord(u'\ufeff')]
        )}

    def fit(self, X, y=None):
        return self

    def clean_(self, soup):
        # Step 1: unescape (twice)
        html_parser = self.html_parser_
        unescaped_soup = html_parser.unescape(html_parser.unescape(soup))

        # Step 2: strip HTML tags
        text = clean_html(unescaped_soup)

        # Step 3: remove zero-width characters
        cleaned = text if text == '' else text.translate(self.normalize_map)

        return cleaned.lower() if self.lowercase else cleaned

    def transform(self, X, y=None):
        column = self.column
        for row in X:
            text = row if self.column is None else row[column]
            yield self.clean_(text)

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

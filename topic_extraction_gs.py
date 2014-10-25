# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

from __future__ import print_function

from argparse import ArgumentParser

import json

from time import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from utils.lfcorpus import get_data_frame
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from operator import itemgetter


# parse commandline arguments
op = ArgumentParser()
op.add_argument("--n_samples", default=1000, type=int,
                help="number of samples to use")
op.add_argument("--n_features", default=1000, type=int,
                help="number of features to expect")
op.add_argument("--n_topics", default=10, type=int,
                help="number of topics to find")
op.add_argument("--n_top_words", default=20, type=int,
                help="number of top words to print")
op.add_argument("--categories", nargs="+", type=str,
                help="number of top words to print")
op.add_argument("--data_dir", type=str,
                help="data directory")
args = op.parse_args()

if args.data_dir is None:
    op.error('Data directory not given')

# Load the 20 newsgroups dataset and vectorize it using the most common word
# frequency with TF-IDF weighting (without top 5% stop words)

t0 = time()
print("Loading dataset and extracting TF-IDF features...")

if args.categories is not None:
    cat_filter = set(args.categories)
else:
    cat_filter = None

dataset = get_data_frame(
    args.data_dir,
    lambda line: json.loads(line)['content'],
    cat_filter=cat_filter)


vectorizer = CountVectorizer(max_df=0.95, max_features=args.n_features,
                                  lowercase=True, stop_words="english")
counts = vectorizer.fit_transform(dataset.data[:args.n_samples])
tfidf = TfidfTransformer(norm="l2", use_idf=True).fit_transform(counts)
print("done in %0.3fs." % (time() - t0))


documents = [nltk.clean_html(document) for document in dataset.data]
stoplist = stopwords.words('english')
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(t) for t in texts]

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

from IPython import embed; embed()
# lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
# lsi.print_topics(20)

lda = models.LdaModel(corpus_tfidf, id2word=dictionary,
                      num_topics=args.n_topics)

for i in range(0, args.n_topics):
    temp = lda.show_topic(i, 10)
    terms = []
    for term in temp:
        terms.append(term[1])
    print("Top 10 terms for topic " + str(i) + ": " + ", ".join(terms))

print('Which LDA topic maximally describes a document?\n')
print('Original document: ' + documents[1])
print('Preprocessed document: ' + str(texts[1]))
print('Matrix Market format: ' + str(corpus[1]))
print('Topic probability mixture: ' + str(lda[corpus[1]]))
print('Maximally probable topic: topic #' + str(max(lda[corpus[1]],
                                                    key=itemgetter(1))[0]))

#
## Fit the model
#print("Fitting the %s model on with n_samples=%d and n_features=%d..."
#      % (args.method, args.n_samples, args.n_features))
#nmf = Decomposition(n_components=args.n_topics).fit(tfidf)
#print("done in %0.3fs." % (time() - t0))
#
## Inverse the vectorizer vocabulary to be able
#feature_names = vectorizer.get_feature_names()
#
#for topic_idx, topic in enumerate(nmf.components_):
#    print("Topic #%d:" % topic_idx)
#    print(" ".join([feature_names[i]
#                    for i in topic.argsort()[:-args.n_top_words - 1:-1]]))
#    print()

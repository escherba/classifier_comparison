from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from operator import itemgetter

documents = [nltk.clean_html(document) for document in nyt_data]
stoplist = stopwords.words('english')
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
# lsi.print_topics(20)

n_topics = 60
lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)

for i in range(0, n_topics):
    temp = lda.show_topic(i, 10)
    terms = []
    for term in temp:
        terms.append(term[1])
    print "Top 10 terms for topic #" + str(i) + ": " + ", ".join(terms)

print 'Which LDA topic maximally describes a document?\n'
print 'Original document: ' + documents[1]
print 'Preprocessed document: ' + str(texts[1])
print 'Matrix Market format: ' + str(corpus[1])
print 'Topic probability mixture: ' + str(lda[corpus[1]])
print 'Maximally probable topic: topic #' + str(max(lda[corpus[1]],
                                                    key=itemgetter(1))[0])

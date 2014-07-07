.PHONY: pca plot plot_py clean extract_topics

CORPUS_DIR=~/dev/py-nlp/var/corpora/livefyre

plot: index.html index.js fit_metrics.csv
	open -a "Safari" $<

plot_py: plot.py fit_metrics.csv
	python $^

pca: pca.py
	python $< \
		--method SVD \
		--vectorizer tfidf \
		--data_dir $(CORPUS_DIR)

extract_topics: topic_extraction.py
	python $< \
		--data_dir $(CORPUS_DIR) \
		--n_samples 10000 \
		--method NMF \
		--n_topics 10 \
		--n_features 4000 \
		--categories spam

fit_metrics.csv: classify.py lf_feat_extract.py lfcorpus_utils.py
	python $< \
		--top_terms 100 \
		--data_dir $(CORPUS_DIR) \
		--output $@

grid_search: grid_search.py
	python $< \
		--scoring f1 \
		--data_dir $(CORPUS_DIR)

clean:
	rm -f *.pyc
	rm fit_metrics.csv

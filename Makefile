.PHONY: pca plot plot_py clean extract_topics env

PYENV = . env/bin/activate;
PYTHON = . env/bin/activate; python
CORPUS_DIR=~/dev/py-nlp/var/corpora/livefyre
CORPUS_DIR2=~/dev/py-nlp/var/corpora/livefyre/dec17
CORPUS_DIR3=~/dev/py-nlp/var/corpora/livefyre/dec29
PLOT_INTERMEDIATE=fit_metrics

env: requirements.txt
	test -d env || virtualenv --no-site-packages env
	$(PYENV) pip install -r requirements.txt
	$(PYENV) pip install matplotlib
	$(PYENV) easy_install ipython

plot: index.html index.js $(PLOT_INTERMEDIATE).csv
	open -a "Safari" $<

plot2: index2.html index2.js fit_metrics2.csv
	open -a "Safari" $<

plot_py: $(PLOT_INTERMEDIATE).png
	open -a "Preview" $^


pca: pca.py
	$(PYTHON) $< \
		--method NMF \
		--vectorizer tfidf \
		--data_dir $(CORPUS_DIR)

extract_topics: topic_extraction.py
	$(PYTHON) $< \
		--data_dir $(CORPUS_DIR) \
		--n_samples 10000 \
		--method NMF \
		--n_topics 10 \
		--n_features 4000 \
		--categories spam

$(PLOT_INTERMEDIATE).png: plot.py $(PLOT_INTERMEDIATE).csv
	$(PYTHON) $^ $@


$(PLOT_INTERMEDIATE).csv: classify.py lf_feat_extract.py lfcorpus_utils.py
	$(PYTHON) $< \
		--vectorizer hashing \
		--top_terms 100 \
		--data_dir $(CORPUS_DIR) \
		--output $@

fit_metrics2.csv: classify.py lf_feat_extract.py lfcorpus_utils.py
	python $< \
		--top_terms 100 \
		--data_test $(CORPUS_DIR3) \
		--data_train $(CORPUS_DIR2) \
		--output $@

grid_search: grid_search.py
	$(PYTHON) $< \
		--scoring f1 \
		--data_dir $(CORPUS_DIR)

clean:
	find . -type f -name "*.pyc" -exec rm -f {} \;
	rm -f fit_metrics.*

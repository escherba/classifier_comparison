.PHONY: pca clean extract_topics env plot_py plot_py2

PYENV = . env/bin/activate;
PYTHON = . env/bin/activate; python
CORPUS_DIR=~/dev/py-nlp/var/corpora/livefyre
CORPUS_DIR2=~/dev/py-nlp/var/corpora/livefyre/dec17
CORPUS_DIR3=~/dev/py-nlp/var/corpora/livefyre/dec29
PLOT_INTERMEDIATE=fit_metrics
PLOT_INTERMEDIATE2=fit_metrics2
OUTPUT=out

env: requirements.txt
	test -d env || virtualenv --no-site-packages env
	$(PYENV) pip install -r requirements.txt
	$(PYENV) pip install matplotlib
	$(PYENV) easy_install ipython

plot_browser: $(PLOT_INTERMEDIATE).browser
plot_browser_time: $(PLOT_INTERMEDIATE2).browser

plot_pylab: $(PLOT_INTERMEDIATE).pylab
plot_pylab_time: $(PLOT_INTERMEDIATE2).pylab

%.browser: $(OUTPUT)/%.scores chart/chart.scpt chart/chart.html chart/chart.js
	osascript chart/chart.scpt "file://"$(CURDIR)"/chart/chart.html#.."/$<

%.pylab: $(OUTPUT)/%.scores.png $(OUTPUT)/%.roc.png
	open -a "Preview" $^

pca: pca.py
	$(PYTHON) $< \
		--method SVD \
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

%.roc.png: plot_roc.py %.roc
	$(PYTHON) $^ $@

%.scores.png: plot_scores.py %.scores
	$(PYTHON) $^ $@

$(OUTPUT)/$(PLOT_INTERMEDIATE).roc $(OUTPUT)/$(PLOT_INTERMEDIATE).scores: %: classify.py utils/feature_extract.py utils/lfcorpus.py $(OUTPUT)
	$(PYTHON) $< \
		--vectorizer tfidf \
		--top_terms 100 \
		--data_dir $(CORPUS_DIR) \
		--output_roc $(basename $@).roc \
		--output     $(basename $@).scores

$(OUTPUT)/$(PLOT_INTERMEDIATE2).roc $(OUTPUT)/$(PLOT_INTERMEDIATE2).scores: %: classify.py utils/feature_extract.py utils/lfcorpus.py $(OUTPUT)
	$(PYTHON) $< \
		--top_terms 100 \
		--vectorizer tfidf \
		--data_test $(CORPUS_DIR3) \
		--data_train $(CORPUS_DIR2) \
		--output_roc $(basename $@).roc \
		--output     $(basename $@).scores

$(OUTPUT):
	mkdir $(OUTPUT)

grid_search: grid_search.py
	$(PYTHON) $< \
		--scoring f1 \
		--data_dir $(CORPUS_DIR)

clean:
	find . -type f -name "*.pyc" -exec rm -f {} \;
	rm -f $(OUTPUT)/*

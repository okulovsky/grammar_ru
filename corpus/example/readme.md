First, we convert the formats of the raw data to the "pseudomarkdown"

* `/raw/book.zip` is a zipped fb2 book, it is converted by `processing_book.py`
* `/raw/medium.hmtl` is an html in the current `medium.com` format. It is converted by notebook `processing_medium.ipynb`

Then, the texts can be converted from the `md` format to the corpus format (`build_corpus.py`)

Additionally, corpus can be featurized to store more data about the text (morphological/syntax analysis etc)



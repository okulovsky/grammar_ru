# Grammar_ru

Initially, `grammar_ru` was envisioned as a set of tools to correct grammar and style errors in Russian texts.

Since the beginning, a lot has changed.

`Grammar_ru` is now a module that governs a representation of language-independent texts in tabular format.

* The texts are processed with tokenization and sentenization, placed in pandas DataFrames and stored in zip-files along
with Table-Of-Contents, or toc-files that contains metadata about each dataframe.
* These zip-files we call corpora.
* Each word, sentence and paragraph receives its unique ID in the corpus
* Relations can be placed in corpus, establishing the relations between fragments of texts
(e.g. that the chapters from translation and original texts are in fact the same chapter).
* `Grammar_ru` also allows you to apply featurizers, such as pymorphy, snowball, slovnet, etc.
* `Grammar_ru` contains useful components to further convert such datasets in torch tensors (based on [Training Grounds Framework](github.com/Outfittery/grounds))

Aside from `grammar_ru`, the repository contains a not-yet-working `app_grammar_ru` which is a docker app that actually 
checks errors in Russian texts. This app is to utilize existing python solutions (pyenchant), as well as ML models,
trained in `grammar_ru` paradygm.

Finally, a creative articulator (ca) project is also temporarily hosted in this repo.


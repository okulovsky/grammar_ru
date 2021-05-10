#!/bin/bash

sudo apt-get install myspell-ru

! wget https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar      # Pretrained word embeddings model weights
! wget https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_ner_news_v1.tar                # Pretrained Named Entity Recognition model weights
! wget https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_morph_news_v1.tar              # Pretrained morphological analyzator model weights 
! wget https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_syntax_news_v1.tar             # Pretrained syntax analyzator model weights

! wget https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.0/lenta-ru-news.csv.gz  # Dataset
! wget https://storage.yandexcloud.net/natasha-nerus/data/nerus_lenta.conllu.gz                     # Training dataset

mv *.tar ./grammar_ru/analyzers/natasha/models
mv *.gz ./grammar_ru/analyzers/natasha/models
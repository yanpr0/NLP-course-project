#!/usr/bin/env bash

git clone https://github.com/RussianNLP/RuCoLA.git

wget https://github.com/dialogue-evaluation/morphoRuEval-2017/raw/master/OpenCorpora_Texts.rar
wget https://github.com/dialogue-evaluation/morphoRuEval-2017/raw/master/GIKRYA_texts_new.zip

unrar e OpenCorpora_Texts.rar
unzip GIKRYA_texts_new.zip

mkdir -p data

python3 build_dataset_parts.py
cat part1.txt part2.txt part3.txt > data/corpora.txt

rm part*
rm unamb_sent_14_6.conllu gikrya_new_train.out gikrya_new_test.out
rm OpenCorpora_Texts.rar GIKRYA_texts_new.zip


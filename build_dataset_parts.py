#!/usr/bin/env python3

from rnnmorph.data_preparation.converter import UDConverter

UDConverter.convert_from_conllu("unamb_sent_14_6.conllu", "part1.txt", with_punct=False, with_forth_column=True)
UDConverter.convert_from_conllu("gikrya_new_train.out", "part2.txt", with_punct=False)
UDConverter.convert_from_conllu("gikrya_new_test.out", "part3.txt", with_punct=False)


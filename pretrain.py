#!/usr/bin/env python3

from rnnmorph.data_preparation.loader import Loader
from rnnmorph.data_preparation.process_tag import process_gram_tag
import json
from transformers import BertTokenizer
from datasets import load_dataset
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
import os


CORPORA_FILE = "data/corpora.txt"
VOCAB_FILE = "data/vocab.txt"
DATASET_FILE = "data/dataset.txt"
SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
VOCAB_JSON = "data/vocab.json"
MODEL_DIR = "data/model"

loader = Loader(language="ru")
loader.parse_corpora([CORPORA_FILE])


with open(VOCAB_FILE, "w") as f:
    print(*SPECIAL_TOKENS, sep='\n', file=f)
    words = set(loader.grammeme_vectorizer_input.name_to_index.keys()) | set(loader.grammeme_vectorizer_output.name_to_index.keys())
    for w in words:
        print(w, file=f)


with open(CORPORA_FILE) as f, open(DATASET_FILE, "w") as d:
    line = ""
    for l in f:
        if l == "\n":
            print(line.rstrip(), file=d)
            line = ""
        else:
            text, lemma, pos_tag, gram = l.strip().split("\t")[0:4]
            gram = process_gram_tag(gram)
            vector_name = pos_tag + '#' + gram
            line += vector_name + ' '


vocab = {}

with open(VOCAB_FILE) as f:
    i = len(vocab)
    for w in f:
        vocab[w.rstrip()] = i
        i += 1

with open(VOCAB_JSON, "w") as f:
    json.dump(vocab, f)



tokenizer = BertTokenizer(VOCAB_FILE, do_lower_case=False, do_basic_tokenize=False)

dataset = load_dataset('text', data_files=[DATASET_FILE])['train']

#comment to use the full dataset :(
dataset = dataset.train_test_split(test_size=0.1)["test"]

dataset = dataset.train_test_split(test_size=0.1)

def encode_with_truncation(examples):
  return tokenizer(examples["text"], truncation=True, padding="max_length",
                   max_length=128, return_special_tokens_mask=True)

train_dataset = dataset["train"].map(encode_with_truncation, batched=True)
test_dataset = dataset["test"].map(encode_with_truncation, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])


model_config = BertConfig(vocab_size=len(tokenizer.vocab), max_position_embeddings=128)
model = BertForMaskedLM(config=model_config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)


os.makedirs(MODEL_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    save_strategy="epoch",
    dataloader_num_workers=2,
    load_best_model_at_end=True,
    learning_rate=1e-5,
    weight_decay=0.0001,
    save_total_limit=1,
    group_by_length=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()


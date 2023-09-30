import os
from argparse import ArgumentParser
from functools import partial
from shutil import rmtree

import numpy as np
from datasets import load_metric
from razdel import tokenize
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    BertModel,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import torch.nn as nn
import torch
from typing import Optional

from RuCoLA.baselines.utils import read_splits

from full_model import BertGram


ACCURACY = load_metric("accuracy", keep_in_memory=True)
MCC = load_metric("matthews_correlation", keep_in_memory=True)
MODEL_TO_HUB_NAME = {
    "rubert-base": "ai-forever/ruBert-base",
    "rubert-large": "ai-forever/ruBert-large",
    "ruroberta-large": "ai-forever/ruRoberta-large",
    "xlmr-base": "xlm-roberta-base",
    "rembert": "google/rembert",
}
MODEL_DIR = "data/model"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

N_SEEDS = 3
N_EPOCHS = 5
LR_VALUES = (1e-5, 3e-5, 5e-5)
DECAY_VALUES = (1e-4, 1e-2, 0.1)
BATCH_SIZES = (32, 64)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    preds = np.argmax(preds, axis=1)

    acc_result = ACCURACY.compute(predictions=preds, references=p.label_ids)
    mcc_result = MCC.compute(predictions=preds, references=p.label_ids)

    result = {"accuracy": acc_result["accuracy"], "mcc": mcc_result["matthews_correlation"]}

    return result


def to_grams(text):
    p = predictor.predict([t.text for t in tokenize(text)])
    return ' '.join([e.pos + '#' + e.tag for e in p if e.pos != 'PUNCT'])


def preprocess_examples(examples, tokenizer, gram_tokenizer):
    result = tokenizer(examples["sentence"], padding=False)

    if "acceptable" in examples:
        result["label"] = examples["acceptable"]

    result['gram_ids'] = gram_tokenizer(list(map(to_grams, examples["sentence"])), padding=True, pad_to_multiple_of=8)['input_ids']

    result["length"] = [len(list(tokenize(sentence))) for sentence in examples["sentence"]]
    return result


MODEL_NAME = "ruroberta-large"


def gram_main(checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_HUB_NAME[MODEL_NAME])

    gram_tokenizer = BertTokenizer(VOCAB_FILE, do_lower_case=False, do_basic_tokenize=False)

    splits = read_splits(as_datasets=True)

    tokenized_splits = splits.map(
        partial(preprocess_examples, tokenizer=tokenizer, gram_tokenizer=gram_tokenizer),
        batched=True,
        remove_columns=["sentence"],
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # seed, lr, wd, bs
    dev_metrics_per_run = np.empty((N_SEEDS, len(LR_VALUES), len(DECAY_VALUES), len(BATCH_SIZES), 2))

    for i, learning_rate in enumerate(LR_VALUES):
        for j, weight_decay in enumerate(DECAY_VALUES):
            for k, batch_size in enumerate(BATCH_SIZES):
                for seed in range(N_SEEDS):
                    main_model = AutoModel.from_pretrained(MODEL_TO_HUB_NAME[MODEL_NAME])

                    model = BertGram(main_model, os.path.join(MODEL_DIR, checkpoint_dir))

                    run_base_dir = f"{MODEL_NAME}_{learning_rate}_{weight_decay}_{batch_size}"

                    training_args = TrainingArguments(
                        output_dir=f"checkpoints/{run_base_dir}",
                        overwrite_output_dir=True,
                        evaluation_strategy="epoch",
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        num_train_epochs=N_EPOCHS,
                        warmup_ratio=0.1,
                        save_strategy="epoch",
                        save_total_limit=1,
                        seed=seed,
                        fp16=True,
                        tf32=False,
                        dataloader_num_workers=4,
                        group_by_length=True,
                        report_to="none",
                        load_best_model_at_end=True,
                        metric_for_best_model="eval_mcc",
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_splits["train"],
                        eval_dataset=tokenized_splits["dev"],
                        compute_metrics=compute_metrics,
                        data_collator=data_collator,
                    )

                    train_result = trainer.train()
                    print(f"{run_base_dir}_{seed}")
                    print("train", train_result.metrics)

                    os.makedirs(f"results/{run_base_dir}_{seed}", exist_ok=True)

                    dev_predictions = trainer.predict(test_dataset=tokenized_splits["dev"])
                    print("dev", dev_predictions.metrics)
                    dev_metrics_per_run[seed, i, j, k] = (
                        dev_predictions.metrics["test_accuracy"],
                        dev_predictions.metrics["test_mcc"],
                    )

                    predictions = trainer.predict(test_dataset=tokenized_splits["test"])

                    np.save(f"results/{run_base_dir}_{seed}/preds.npy", predictions.predictions)

                    #rmtree(f"checkpoints/{run_base_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-dir", required=True)
    args = parser.parse_args()
    gram_ main(args.model_name)


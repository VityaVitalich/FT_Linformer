import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys

import argparse
import importlib

parser = argparse.ArgumentParser(description="Example script for argument parsing")
parser.add_argument("-d", "--dataset", type=str, help="name of dataset")
args = parser.parse_args()

cur_config = importlib.import_module("configs." + args.dataset)

# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
#     map(str, cur_config.cuda_devices)
# )

import datasets
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import torch
import numpy as np

print(torch.cuda.device_count())


def create_data(dataset, tokenizer):
    train_set = datasets.load_dataset(dataset, split='train').remove_columns(['idx'])
    val_set = datasets.load_dataset(dataset, split='validation').remove_columns(['idx'])

    dynamic_padding = True

    def tokenize_func(examples):
        return tokenizer(examples["sentence"], truncation=True)#,  max_length=512,  padding=True

    encoded_dataset_train = train_set.map(tokenize_func, batched=True)
    encoded_dataset_test = val_set.map(tokenize_func, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    return encoded_dataset_train, encoded_dataset_test, data_collator

if __name__ == '__main__':

    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=cur_config.num_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    encoded_dataset_train, encoded_dataset_test, data_collator = create_data(args.dataset, tokenizer)

    metric = datasets.load_metric('accuracy')
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=cur_config.training_args,
        train_dataset=encoded_dataset_train,
        eval_dataset=encoded_dataset_test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print(trainer.state.best_metric, trainer.state.best_model_checkpoint)
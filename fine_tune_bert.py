import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from transformers.models.bert.modeling_linbert import BertForSequenceClassification, BertForMaskedLM
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForLanguageModeling

import torch
import numpy as np

print(torch.cuda.device_count())

def create_mlm_data(tokenizer):
    name = 'rotten_tomatoes'
    train_set = datasets.load_dataset('imdb', split='train').remove_columns(['label'])
    val_set = datasets.load_dataset('imdb', split='test').remove_columns(['label'])
    # train_set = datasets.load_dataset(name,  split='train').remove_columns(['label'])
    # val_set = datasets.load_dataset(name,  split='validation').remove_columns(['label'])

    def tokenize_func(examples):
        return tokenizer(examples["text"], max_length=cur_config.max_len, padding=cur_config.padding_type, truncation=True)

    encoded_dataset_train = train_set.map(tokenize_func, batched=True)
    encoded_dataset_val = val_set.map(tokenize_func, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer)

    return encoded_dataset_train, encoded_dataset_val, data_collator

def create_data(dataset, tokenizer):
    train_set = datasets.load_dataset(dataset, split='train').remove_columns(['idx'])
    val_set = datasets.load_dataset(dataset, split='validation').remove_columns(['idx'])

    dynamic_padding = True

    def tokenize_func(examples):
        return tokenizer(examples["sentence"], max_length=cur_config.max_len, padding=cur_config.padding_type, truncation=True)

    encoded_dataset_train = train_set.map(tokenize_func, batched=True)
    encoded_dataset_test = val_set.map(tokenize_func, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    return encoded_dataset_train, encoded_dataset_test, data_collator

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

    for layer in model.bert.encoder.layer:
        layer.attention.self.E.requires_grad = True
        layer.attention.self.F.requires_grad = True

if __name__ == '__main__':

    model = BertForSequenceClassification.from_pretrained(cur_config.model_name, num_labels=cur_config.num_labels)
    tokenizer = BertTokenizer.from_pretrained(cur_config.model_name)

    if cur_config.linearize:
        model.linearize(cur_config.max_len, cur_config.k)

    encoded_dataset_train, encoded_dataset_test, data_collator = create_data(args.dataset, tokenizer)

    metric = datasets.load_metric('accuracy')
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)
    
    if cur_config.freeze:
        freeze(model)
        
    if cur_config.pre_training:
        mlm_model = BertForMaskedLM.from_pretrained(cur_config.model_name)
        mlm_model.linearize(cur_config.max_len, cur_config.k)
        if cur_config.freeze:
            freeze(mlm_model)

        mlm_train_set, mlm_val_set, mlm_collator = create_mlm_data(tokenizer)

        trainer = Trainer(
            model=mlm_model,
            args=cur_config.pre_training_args,
            train_dataset=mlm_train_set,
            eval_dataset=mlm_val_set,
            data_collator=mlm_collator,
        )
        trainer.train()
        for parameter_lm, parameter_cl in zip(mlm_model.bert.parameters(), model.bert.parameters()):
            parameter_cl.data = parameter_lm.data
            #print(parameter_lm.data.size() == parameter_cl.data.size())


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
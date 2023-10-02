from transformers import TrainingArguments

num_labels = 2
max_len = 128
padding_type = 'max_length'
model_name = 'VityaVitalich/bert-base-cased-sst2' #'VityaVitalich/bert-tiny-sst2' #'bert-base-cased'#'M-FAC/bert-tiny-finetuned-sst2'

linearize = True
k = 16
pre_training = True
freeze = True
cuda_devices = 3

training_args = TrainingArguments(
    output_dir='/home/FT_Linformer/experiments/sst/notebooks/results/',
    learning_rate=3e-4,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=10,
    weight_decay=0.1,
    metric_for_best_model="accuracy",
    fp16=False,
    fp16_full_eval=False,
    evaluation_strategy="steps",
    eval_steps=200,
    seed=42,
    logging_strategy="steps",
    report_to="all",
    logging_steps=200
)

pre_training_args = TrainingArguments(
    output_dir = '/home/FT_Linformer/experiments/sst/notebooks/results/',
    learning_rate=3e-4,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=15,
    weight_decay=0.1,
    fp16=False,
    fp16_full_eval=False,
    evaluation_strategy="epoch",
  #  eval_steps=500,
    seed=42,
    save_strategy = "epoch",
    logging_strategy="epoch",
   # logging_steps=500
)
from transformers import TrainingArguments

num_labels = 2
max_len = 128
padding_type = 'longest'
model_name = 'bert-base-cased' #'VityaVitalich/bert-base-cased-sst2' #'VityaVitalich/bert-tiny-sst2' #'M-FAC/bert-tiny-finetuned-sst2'

linearize = False
k = 32
pre_training = True
freeze = False
cuda_devices = 1

training_args = TrainingArguments(
    output_dir='/home/FT_Linformer/experiments/sst/notebooks/results/',
    learning_rate=3e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=5,
    weight_decay=0.1,
    metric_for_best_model="accuracy",
    fp16=True,
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
    learning_rate=3e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=15,
    weight_decay=0.1,
    fp16=True,
    fp16_full_eval=False,
    evaluation_strategy="epoch",
  #  eval_steps=500,
    seed=42,
    save_strategy = "epoch",
    logging_strategy="epoch",
   # logging_steps=500
)
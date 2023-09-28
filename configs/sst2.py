from transformers import TrainingArguments

num_labels = 2

cuda_devices = [
    1
]

training_args = TrainingArguments(
    output_dir='/home/FT_Linformer/experiments/sst/notebooks/results/',
    learning_rate=3e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=5,
    weight_decay=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="epoch",
  #  eval_steps=500,
    seed=42,
    save_strategy = "epoch",
    save_total_limit=5,
    logging_strategy="epoch",
    report_to="all",
   # logging_steps=500
)
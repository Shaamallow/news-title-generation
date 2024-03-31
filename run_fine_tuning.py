"""Fine-tune the t5_base_fr_sum_cnndm model on this task."""

from transformers import (
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
)

from src.load_models import t5_base_fr_sum_cnndm, device
from src.load_data import load_data, preprocess_from_df
from src.metrics import gen_compute_metrics

BATCH_SIZE = 1
DEVICE = device()


model, tokenizer = t5_base_fr_sum_cnndm()
model = model.to(DEVICE)

train_df, validation_df, _ = load_data()
train_dataset = preprocess_from_df(train_df, tokenizer)
validation_dataset = preprocess_from_df(validation_df, tokenizer)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="../outputs/t5_base_fr_sum_cnndm_finetuned",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    num_train_epochs=1,
    eval_steps=2,
    save_steps=2,
    warmup_steps=1,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,
)

compute_metrics = gen_compute_metrics(tokenizer)

# Construct the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,  # type: ignore
    eval_dataset=validation_dataset,  # type: ignore
)


trainer.train()

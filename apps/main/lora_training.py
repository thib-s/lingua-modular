from transformers import Trainer, TrainingArguments

from peft import LoraConfig, TaskType
from peft import get_peft_model

from apps.main.save_to_hf import load_consolidated_model
model, model_cfg = load_consolidated_model("/lustre/fswork/projects/rech/reh/commun/dumps/modular_base/checkpoints/0000038000/consolidated")

# peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
peft_config = LoraConfig(
    target_modules=["wq", "wv"],
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir="/lustre/fswork/projects/rech/reh/commun/mt0-large-lora",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("output_dir")
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer

from datasets import load_dataset

from peft import LoraConfig
from peft import get_peft_model

from apps.main.save_to_hf import load_consolidated_model

# Load the dataset
# pubmed_path = "/lustre/fsmisc/dataset/HuggingFace/pubmed"
# dataset = load_dataset(pubmed_path, split="train")
# dataset = dataset.train_test_split(test_size=0.1)
from datasets import load_dataset
kwargs = dict(split="train")
dataset = load_dataset("/lustre/fsmisc/dataset/HuggingFace/OpenLLM-France/Lucie-Training-Dataset", "Eurovoc", **kwargs)
dataset = dataset.train_test_split(test_size=0.1)

# Load the tokenizer
tokenizer_path = "/lustre/fswork/projects/rech/reh/commun/tokenizers/llama3/original"

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

# def preprocess(example):
#     encoding = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
#     encoding["labels"] = encoding["input_ids"].copy()
#     return encoding

# tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

def tokenize_batch_collate_fn(batch):
    texts = [example["text"] for example in batch]
    encodings = tokenizer(
        texts,
        padding="longest",  # or "max_length"
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return {
        "token_values": encodings["input_ids"],
        "target": encodings["input_ids"].clone()
    }

model, model_cfg = load_consolidated_model("/lustre/fswork/projects/rech/reh/commun/dumps/modular_base/checkpoints/0000038000/consolidated")

# peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
peft_config = LoraConfig(
    target_modules=["wq", "wv"],
)
model = get_peft_model(model, peft_config)

output_dir = "/lustre/fswork/projects/rech/reh/commun/FOR-sight-ai/lingua-modular-eurovoc"
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=False,
)

def compute_loss_func(output):
    return output

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
    data_collator=tokenize_batch_collate_fn,
    compute_loss_func=compute_loss_func
)

trainer.train()

model.save_pretrained(output_dir)
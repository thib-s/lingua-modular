import os
import math
import argparse

import torch

from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset

from peft import LoraConfig, TaskType
from peft import get_peft_model

os.environ["WANDB_MODE"] = "offline"

MODULE = {
    "Pile-Freelaw": 14 * 1e9
}

def prepare_module_ds(module_name):
    if module_name == "Pile-Freelaw":
        # kwargs = dict(split="train", streaming=True)
        kwargs = dict(split="train")
        dataset = load_dataset("/lustre/fsmisc/dataset/HuggingFace/OpenLLM-France/Lucie-Training-Dataset", "Pile-FreeLaw", **kwargs)
        return dataset
    else:
        raise ValueError(f"Unknown module name: {module_name}")

def tokenize_batch_collate_fn(batch):
    texts = [example["text"] for example in batch]
    encodings = tokenizer(
        texts,
        padding="longest",         # or "max_length" if you prefer
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )
    encodings["labels"] = encodings["input_ids"].clone()
    return {k: v.contiguous() for k, v in encodings.items()}

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--module_name", type=str, default="Pile-Freelaw")
    argparser.add_argument("--nb_gpus", type=int, default=4)
    argparser.add_argument("--batch_size", type=int, default=4)

    args = argparser.parse_args()
    NB_GPUS = args.nb_gpus
    NB_TOKENS_MODULE = MODULE[args.module_name]
    BATCH_SIZE = args.batch_size
    MAX_STEPS = math.ceil(1.1 * NB_TOKENS_MODULE/ (NB_GPUS * 2048 * BATCH_SIZE))

    output_dir = f"/lustre/fswork/projects/rech/reh/commun/FOR-sight-ai/SmolLM2-1.7B/{args.module_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    dataset = prepare_module_ds(args.module_name)

    # Load the tokenizer
    checkpoint = "/lustre/fswork/projects/rech/reh/commun/models/SmolLM2-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the pre-trained model
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    # LoRA preparation
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"], # "k_proj", \"gate_proj\", \"up_proj\", \"down_proj\"\n",
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="steps",
        logging_steps=50,
        save_steps=50,
        load_best_model_at_end=False,
        # remove_unused_columns=False,
        bf16=True,
        max_steps=MAX_STEPS,
        disable_tqdm=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=tokenize_batch_collate_fn,
    )
    # Start training
    trainer.train()

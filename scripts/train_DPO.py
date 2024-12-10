import os
import sys
os.sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig
from huggingface_hub import login
from utils.data_utils import prepare_data,load_dataset_config,format_instruction_prompt
import json
import argparse
HF_TOKEN = "hf_XXX"
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-7B")
parser.add_argument("--data_config_path", type=str, default="configs/datasets.yaml")
parser.add_argument("--hf_upload_name", type=str, default="overfit-brothers/Qwen2.5-7B-SFT")
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_seq_length", type=int, default=1024)
parser.add_argument("--warmup_ratio", type=float, default=0.05)
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--learning_rate", type=float, default=6e-6)
parser.add_argument("--output_dir", type=str, default="outputs")
parser.add_argument("--log_dir", type=str, default="runs/logs")
parser.add_argument("--DPO_mode", type=str, default="offline")
def generate_prompt_DPO(example, tokenizer):
    EOS_TOKEN = tokenizer.eos_token
    prompt = example['prompt']
    rejected = example['rejected']
    chosen = example['chosen']

    example['prompt'] = prompt + EOS_TOKEN
    example['rejected'] = rejected + EOS_TOKEN
    example['chosen'] = chosen + EOS_TOKEN

    example['token_length'] = max([tokenizer.tokenize(example['prompt']+example['chosen'],example['prompt']+example['rejected'])])
    return example

def DPO(args):
    login(HF_TOKEN)
    

    
    output_dir = os.path.join(args.output_dir, args.hf_upload_name.split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)
    
    lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=bnb_config)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset_list = load_dataset_config(config_path=args.data_config_path)
    train_dataset = prepare_data(dataset_list,do_interleave=False, mode="DPO")
    
    train_dataset = train_dataset.map(generate_prompt_DPO, fn_kwargs={"tokenizer": tokenizer}, num_proc=8)
    train_dataset = train_dataset.filter(lambda x:len(x["token_length"]) < args.max_seq_length-2)
    training_args = DPOConfig(
        output_dir=args.output_dir,
        logging_dir=args.log_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=5,
        bf16=True,
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=0.1,
        peft_config=lora_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,

    )
    trainer.train()
    os.makedirs(args.log_dir, exist_ok=True)

    # 로깅 정보를 하나의 파일에 저장
    log_info = {
        "args_info": vars(args),
        "dataset_list": dataset_list,
    }
    log_file_name = f"{args.hf_upload_name.split('/')[-1]}_info.json"
    with open(os.path.join(args.log_dir, log_file_name), "w") as f:
        json.dump(log_info, f, indent=4)
    # LoRA Adapter 저장
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.push_to_hub(args.hf_upload_name, private=True)
    tokenizer.push_to_hub(args.hf_upload_name, private=True)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.DPO_mode == "offline":
        DPO(args)

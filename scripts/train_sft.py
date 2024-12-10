import os
os.sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from datasets import load_dataset,Dataset
from utils.data_utils import prepare_data,load_dataset_config,format_instruction_prompt
from huggingface_hub import login
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
HF_TOKEN = "hf_XXXXX"
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-7B")
parser.add_argument("--data_config_path", type=str, default="configs/datasets.yaml")
parser.add_argument("--hf_upload_name", type=str, default="overfit-brothers/Qwen2.5-7B-SFT")
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_seq_length", type=int, default=1024)
parser.add_argument("--warmup_ratio", type=float, default=0.05)
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=6e-6)
parser.add_argument("--output_dir", type=str, default="outputs")
parser.add_argument("--log_dir", type=str, default="runs/logs")
args = parser.parse_args()

def collate(elements, tokenizer):
    # used in case of using normal "Trainer" instead of "SFTTrainer"
    tokenlist = [e["input_ids"] for e in elements]
    max_len = max(len(t) for t in tokenlist)

    input_ids, labels, attention_masks = [], [], []
    for tokens in tokenlist:
        pad_len = max_len - len(tokens)
        input_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
        labels.append(tokens + [-100] * pad_len)
        attention_masks.append([1] * len(tokens) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),  # input_ids는 정수형이므로 변경하지 않습니다
        "labels": torch.tensor(labels, dtype=torch.long),  # labels도 정수형입니다
        "attention_mask": torch.tensor(attention_masks, dtype=torch.bfloat16)  # attention_mask만 bf16으로 변환
    }
    
    
def main():
    login(token = HF_TOKEN)
    # attn_impl = "flash_attention_2"
    attn_impl = "eager" if "gemma" in args.model_name else "flash_attention_2"
    try:
        model = AutoLigerKernelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map="auto"
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    print("\033[91muse transformers model\033[0m")
        
    dataset_list = load_dataset_config(config_path=args.data_config_path)

    train_dataset = prepare_data(dataset_list,do_interleave=False)
    print(f"정제 전 데이터셋 길이 : \033[91m{len(train_dataset)}\033[0m")

    train_dataset = train_dataset.map(format_instruction_prompt,fn_kwargs={"tokenizer": tokenizer})
    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], truncation=True, max_length=args.max_seq_length)
        
        # 모든 시퀀스의 마지막 토큰을 한 번에 EOS 토큰으로 설정
        input_ids = tokens['input_ids']
        for seq in input_ids:
            seq[-1] = tokenizer.eos_token_id
            
        return tokens

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
    print(f"정제 후 데이터셋 길이 : \033[91m{len(tokenized_dataset)}\033[0m")


    output_dir = os.path.join(args.output_dir, args.hf_upload_name.split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)
    training_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        bf16 = True,
        logging_steps=5,
        optim="paged_adamw_32bit",
        lr_scheduler_type="linear",
        push_to_hub=True,
        hub_model_id=args.hf_upload_name,
        seed=42,
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        use_liger=True,
        hub_token=HF_TOKEN,
        hub_private_repo=True,
        # resume_from_checkpoint=True   
    )
    
    # 학습 전에 TrainingArguments 출력
    print("\033[94m===== TrainingArguments =====\033[0m")
    specified_args = [
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "warmup_ratio",
        "optim",
        "num_train_epochs",
        "learning_rate",
        "max_seq_length",
        "hf_upload_name"
    ]
    for arg in specified_args:
        if arg == "hf_upload_name":
            print(f"\033[94m{arg}: {training_args.hub_model_id}\033[0m")
        else:
            print(f"\033[94m{arg}: {getattr(training_args, arg)}\033[0m")
    print("\033[94m=============================\033[0m")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        dataset_num_proc=16,
        # data_collator=DataCollatorForCompletionOnlyLM
        data_collator=lambda x: collate(x, tokenizer),
        packing=False,
    )
    
    train_stats = trainer.train()
        
    # 로깅 디렉토리 생성
    os.makedirs(args.log_dir, exist_ok=True)

    # 로깅 정보를 하나의 파일에 저장
    log_info = {
        "args_info": vars(args),
        "dataset_list": dataset_list
    }
    log_file_name = f"{args.hf_upload_name.split('/')[-1]}_info.json"
    with open(os.path.join(args.log_dir, log_file_name), "w") as f:
        json.dump(log_info, f, indent=4)
        
if __name__ == "__main__":
    main()

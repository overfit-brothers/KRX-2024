from datasets import load_dataset,Dataset,concatenate_datasets,interleave_datasets
from typing import List, Dict, Any
from constants.prompts import get_alpaca_prompt
import yaml
from pathlib import Path

HF_TOKEN = "hf_XXXX"
def _rename_columns(example: Dataset) -> Dataset:
    """
    Dataset 클래스 객체를 입력으로 받아 train 스플릿이 있으면 추출하고, 칼럼 명도 표준화해서 반환합니다.
    """
    if 'train' in example:
        example = example['train']
    
    existing_columns = example.column_names
    rename_map = {
        "prompt": "question",
        "prompts": "question",
        "questions": "question",
        "instruction": "question",
        "response": "answer",
        "responses": "answer",
        "answers": "answer",
        "output": "answer",
        "outputs": "answer"
    }
    
    rename_map = {k: v for k, v in rename_map.items() if k in existing_columns}
    example = example.rename_columns(rename_map)
    
    keep_columns = ["question", "answer"]
    example = example.select_columns(keep_columns)
    
    return example

def load_dataset_config(config_path='configs/datasets.yaml'):
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['datasets']

def prepare_data(example:List[Any], do_interleave=False,mode="default",*args, **kwargs):
    """
    여러 데이터셋을 하나로 합쳐 반환하는 함수
    """
    dataset_list = []
    for dataset_name in example:
        dataset = load_dataset(dataset_name,token=HF_TOKEN)
        if mode == "default":
            dataset = _rename_columns(dataset)
            dataset_list.append(dataset)
        else:
            dataset_list.append(dataset['train'])            
    
    if do_interleave:
        return_dataset = interleave_datasets(dataset_list,stopping_strategy="all_exhausted")
        
    return_dataset = concatenate_datasets(dataset_list).shuffle(seed=42)
    
    return return_dataset

def load_multi_config(config_path='configs/datasets.yaml'):
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    mcqa = config['MCQA']
    instruct = config['instruct']
    return {'MCQA':mcqa,'instruct':instruct}


# def format_prompt(examples, tokenizer):
#     """
#     알파카 프롬프트에 맞추어 포맷팅하는 함수
#     """
#     prompt_format = get_alpaca_prompt()
#     EOS_TOKEN = tokenizer.eos_token

#     question = examples['question']
#     answer = examples['answer']
#     texts = []
    
#     for q,a in zip(question, answer):
#         text = prompt_format.format(q,a)+EOS_TOKEN
#         texts.append(text)
#     return {'text':texts,}

def format_instruction_prompt(examples, tokenizer):
    """
    인스트럭션 템플릿에 맞추어 포맷팅하는 함수
    """
    # TODO: 시스템 프롬프트 처리
    question = examples['question']
    answer = examples['answer']    
    user = {
        "role": "user",
        "content": question
    }
    assistant = {
        "role": "assistant",
        "content": answer
    }
    
    chat = [user, assistant]
    template = tokenizer.apply_chat_template(chat, tokenize=False)
    if tokenizer.eos_token not in template[-100:]:
        template+=tokenizer.eos_token
    return {'text':template}

from metrics import Trec
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from random import randrange
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from collections import defaultdict
import sys
import os
# Set batch size and other relevant parameters
batch_size = 1
checkpoint_dir = sys.argv[1]

#checkpoint_dir = 'meta-llama/Llama-2-7b-chat-hf'
dataset_name = 'data/msmarco/dl2020_54.bm25.passage.hf'
#dataset_name = 'data/msmarco/dl2020_single.bm25.passage.hf'

output_dir = checkpoint_dir + dataset_name.split('/')[-1]+'/run.no_format_prmpt'
os.makedirs(output_dir, exist_ok=True)
use_flash_attention = True
load_unmerged = True

qrels_file = 'data/msmarco/2020qrels-pass.txt'




def format_instruction(sample):
    return f"""### Instruction:
Write a question that this passsage could answer.
### Passage:
{sample['document']}
### Question:
{sample['query']}"""

def format_instruction(sample):
    return f"write a question that this passsage could answer.\npassage:\n{sample['document']}\nquestion:\n{sample['query']}"


if use_flash_attention:
    # unpatch flash attention
    from llama_patch import unplace_flash_attn_with_attn
    unplace_flash_attn_with_attn()

tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, padding_side='right')
tokenizer.add_special_tokens({"pad_token":"<pad>"})
#load unmerged
if load_unmerged:
    # load base LLM model and tokenizer
    unmerged_model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        #load_in_4bit=True,
        device_map='auto'        
    )
    model = unmerged_model.merge_and_unload()
    # Save the merged model
    #model.save_pretrained(checkpoint_dir + '_merged', safe_serialization=True)
    #tokenizer.save_pretrained(checkpoint_dir+'_merged')
else:
    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_use_dobule_quant=False
    )
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, device_map='auto', quantization_config=quant_config)

model.config.pretraining_tp = 1
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))
model.model.embed_tokens.padding_idx = len(tokenizer) - 1
model.model.embed_tokens._fill_padding_idx_with_zero()

model.config.use_cache = False


dataset = Dataset.load_from_disk(dataset_name)
sample = dataset[randrange(len(dataset))]
prompt = format_instruction(sample)

def remove_token_type_ids(inp):
    if 'token_type_ids' in inp:
        del inp['token_type_ids']


def collate_fn(batch):
    qids = [sample['qid'] for sample in batch]
    dids = [sample['did'] for sample in batch]
    target = [sample['query'] for sample in batch]
    instr = [format_instruction(sample)  for sample in batch]  # Add prompt to each text
    instr_tokenized = tokenizer(instr, padding=True, truncation=True, return_tensors="pt")
    target_tokenized = tokenizer(target, padding=True, truncation=True, return_tensors="pt")
    remove_token_type_ids(instr_tokenized)
    remove_token_type_ids(target_tokenized)
    return qids, dids, instr_tokenized, target_tokenized



# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)


def get_scores(model, instr_tokenized, target_tokenized):
    remove_token_type_ids(instr_tokenized)
    logits = model(**instr_tokenized.to('cuda')).logits
    loss_fct = CrossEntropyLoss(reduction='none', ignore_index=model.config.pad_token_id)
    target = target_tokenized.input_ids.to('cuda')[..., 2:]
    logits_target = logits[:, -(target_tokenized.input_ids.shape[1] -1):-1, :].permute(0, 2, 1)
    #print(logits_target)
    #mask = (instr_tokenized.input_ids == model.config.eos_token_id).cumsum(dim=1) == 0
    loss = loss_fct(logits_target, target)
    return -torch.exp(loss.mean(1).unsqueeze(1))




res_test = defaultdict(dict)
with torch.inference_mode():
    for batch_inp in tqdm(dataloader): 
        qids, dids, instr, target = batch_inp
        instr_tokenized= instr.to('cuda')
        target_tokenized = target.to('cuda')
        scores = get_scores(model, instr_tokenized, target_tokenized)
        batch_num_examples = scores.shape[0]
        # for each example in batch
        for i in range(batch_num_examples):
            res_test[qids[i]][dids[i]] = scores[i].item()

    sorted_scores = []
    q_ids = []
    # for each query sort after scores
    for qid, docs in res_test.items():
        sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get
, reverse=True)]
        q_ids.append(qid)
        sorted_scores.append(sorted_scores_q)


test = Trec('ndcg_cut_10', 'trec_eval', qrels_file, 1000, ranking_file_path=output_dir)
eval_score = test.score(sorted_scores, q_ids)
#print(eval_score)

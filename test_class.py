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
dataset_name = 'data/msmarco/dl2020_54.bm25.passage.ref.hf'
#dataset_name = 'data/msmarco/dl2020_single.bm25.passage.hf'

output_dir = checkpoint_dir + dataset_name.split('/')[-1] +'/run'
os.makedirs(output_dir, exist_ok=True)
use_flash_attention = True
load_unmerged = True

qrels_file = 'data/msmarco/2020qrels-pass.txt'


def format_instruction(sample):
    return f"write a question that this passsage could answer.\npassage:\n{sample['document']}\nquestion:\n{sample['query']}"


def format_instruction(sample):
    q = sample['query']
    p = sample['document']
    return f"Does the passage answer the query?\nQuery: '{q}'\nPassage: '{p}'\nAnswer:"


#Passage1 is the perfect answer to the query. Does Passage2 answer the query better as Passage1?
def format_instruction(sample):
    return f"""### Instruction:
Does Passage 2 answer the query better than Passage 1? yes or no?
### Query:
{sample['query']}
### Passage #1:
{sample['gen_rel_document']}
### Passage #2:
{sample['document']}
### Answer:
"""
def format_instruction(sample):
    return f"""### Instruction:
Does the passage answer the query?
### Query:
{sample['query']}
### Passage:
{sample['document']}
### Answer:
"""
if use_flash_attention:
    # unpatch flash attention
    from llama_patch import unplace_flash_attn_with_attn
    unplace_flash_attn_with_attn()

print(checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained("llama_models/llama-70-chat-int4-msmarco-rank_all_modules_response_only_class/checkpoint-50/")
#tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.padding_side = 'right'
#load unmerged
if load_unmerged:
    # load base LLM model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        #load_in_4bit=True,
        device_map='auto'        
    )
    model = model.merge_and_unload()
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

    if use_flash_attention:
        from llama_patch import upcast_layer_for_flash_attention
        torch_dtype = torch.float16
        #torch_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
        model = upcast_layer_for_flash_attention(model, torch_dtype)

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

True_tokenid = tokenizer.encode('True', add_special_tokens=False)[-1]
False_tokenid = tokenizer.encode('False', add_special_tokens=False)[-1]
true_tokenid = tokenizer.encode('true', add_special_tokens=False)[-1]
false_tokenid = tokenizer.encode('false', add_special_tokens=False)[-1]
yes_tokenid = tokenizer.encode('yes', add_special_tokens=False)[-1]
no_tokenid = tokenizer.encode('no', add_special_tokens=False)[-1]
Yes_tokenid = tokenizer.encode('Yes', add_special_tokens=False)[-1]
No_tokenid = tokenizer.encode('No', add_special_tokens=False)[-1]
y_tokenid = tokenizer.encode('y', add_special_tokens=False)[-1]
n_tokenid = tokenizer.encode('n', add_special_tokens=False)[-1]

false_tokenid = 4541
true_tokenid = 3009
def get_scores(model, instr_tokenized, target_tokenized, print_bool):
    remove_token_type_ids(instr_tokenized)
    scores = model.generate(**instr_tokenized.to('cuda'), max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True).scores
    scores = torch.stack(scores)
    if print_bool:
        print('max prob token', tokenizer.batch_decode(scores.max(2).indices), scores.max(2).indices.ravel().item() ,scores.max(2).values.ravel().item())
        print(scores[0, :, [True_tokenid, False_tokenid, true_tokenid, false_tokenid, Yes_tokenid, No_tokenid, yes_tokenid, no_tokenid]])
    scores = scores[0, :, [false_tokenid, true_tokenid]].float()
    true_prob = torch.softmax(scores, 1)[:, 1]
    return true_prob

    



steps = 0
res_test = defaultdict(dict)
with torch.inference_mode():
    for batch_inp in tqdm(dataloader): 
        qids, dids, instr, target = batch_inp
        instr_tokenized= instr.to('cuda')
        target_tokenized = target.to('cuda')
        scores = get_scores(model, instr_tokenized, target_tokenized, steps<=100)
        batch_num_examples = scores.shape[0]
        # for each example in batch
        for i in range(batch_num_examples):
            res_test[qids[i]][dids[i]] = scores[i].item()
        steps += 1
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
print(eval_score)

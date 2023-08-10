
from trl import SFTTrainer
from random import randrange
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb
def format_instruction(sample):
    return f"""### Instruction:
Write a question that this passsage could answer.
### Passage:
{sample['positive_document']}
### Question:
{sample['query']}"""

#def format_instruction(sample):
#    return f"""{sample['positive_document']}<eos>{sample['query']}"""

dataset = Dataset.load_from_disk('data/msmarco/msmarco.10m.hf')

print(format_instruction(dataset[randrange(len(dataset))]))

use_flash_attention = True
# COMMENT IN TO USE FLASH ATTENTION
# replace attention with flash attention
if torch.cuda.get_device_capability()[0] >= 8:
    from llama_patch import replace_attn_with_flash_attn
    print("Using flash attention")
    replace_attn_with_flash_attn()
    use_flash_attention = True


# Hugging Face model id
#model_id = "NousResearch/Llama-2-7b-hf" # non-gated
#model_id = "meta-llama/Llama-2-7b-chat-hf" # gated
model_id = "meta-llama/Llama-2-7b-hf" # gated
#model_id = "meta-llama/Llama-2-13b-chat-hf" # gated
#model_id = "meta-llama/Llama-2-70b-chat-hf" # gated

cache_dir = '/scratch-shared/drautmp/transformers_cache/'

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.padding_side = "right"


# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache=False, device_map="auto", cache_dir=cache_dir)
model.config.pretraining_tp = 1
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))
model.model.embed_tokens.padding_idx = len(tokenizer) - 1
model.model.embed_tokens._fill_padding_idx_with_zero()

# Validate that the model is using flash attention, by comparing doc strings
if use_flash_attention:
    from llama_patch import forward
    assert model.model.layers[0].self_attn.forward.__doc__ == forward.__doc__, "Model is not using flash attention"




def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    #cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    print('lora modules', lora_module_names)
    return list(lora_module_names)





# prepare model for training
model = prepare_model_for_kbit_training(model)
target_modules = find_all_linear_names(model)


# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
)



model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())

args = TrainingArguments(
    output_dir="llama_models/llama-7-int4-msmarco-rank_all_modules",
    max_steps=5000,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    #disable_tqdm=True # disable tqdm since with packing values are in correct
)




max_seq_length = 512 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
)

# train
trainer.train() # there will not be a progress bar since tqdm is disabled

# save model
trainer.save_model()

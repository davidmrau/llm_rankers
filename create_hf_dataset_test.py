import pandas as pd
import json
from datasets import Dataset, Value


def load_mapping_file(mapping_file):
    # Read mapping file and create a dictionary with IDs as keys and texts as values
    mapping_df = pd.read_csv(mapping_file, sep="\t", names=["id", "text"], dtype='str')
    mapping_dict = dict(zip(mapping_df["id"], mapping_df["text"]))
    return mapping_dict

def create_memory_friendly_dataset(query_mapping_file, document_mapping_file, gen_rel_doc_file, trec_run_file):
    # Load query and document mappings into dictionaries
    query_mapping = load_mapping_file(query_mapping_file)
    document_mapping = load_mapping_file(document_mapping_file)
    gen_rel_document_mapping = load_mapping_file(gen_rel_doc_file)
    
    print(gen_rel_document_mapping)
    # Read TREC run file and process it into a list of dictionaries
    trec_run_data = []
    with open(trec_run_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            qid = parts[0]
            did = parts[2]
            trec_run_data.append((qid, did))
    
    # Organize the data into a dictionary format
    data = {"query": [], "document": [], "did": [], "qid": [], "gen_rel_document": []}
    for (qid, did) in trec_run_data:
        
        # Look up query and document using dictionaries
        if qid in query_mapping and did in document_mapping and qid in gen_rel_document_mapping:
            query = query_mapping[qid]
            document = document_mapping[did]
            gen_rel_doc = gen_rel_document_mapping[qid]
            data["qid"].append(qid) 
            data["did"].append(did) 
            data["query"].append(query)
            data["document"].append(document)
            data["gen_rel_document"].append(gen_rel_doc)
        else:
            print(qid, did)
    
    # Create a Hugging Face Dataset object
    memory_friendly_dataset = Dataset.from_dict(data)
    
    return memory_friendly_dataset


config = json.loads(open('config.json').read())
query_file = config['test']['2020_pass']['queries']
document_file = config['train']['pass']['docs']
#triples_file = config['train']['pass']['triples']
trec_run_file = 'data/msmarco/run.msmarco-passage.bm25.topics.dl20_54.txt_top_100'
gen_rel_doc_file = "data/msmarco/chat-gpt.msmarco-passagetest2020_54.tsv"

dataset = create_memory_friendly_dataset(query_file, document_file, gen_rel_doc_file, trec_run_file)


dataset.save_to_disk("data/msmarco/dl2020_single.bm25.passage.ref.hf")


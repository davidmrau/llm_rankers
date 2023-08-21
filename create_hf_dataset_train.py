import pandas as pd
from datasets import Dataset, Value

import json 
def triples_generator():
    batch_size = 100000
    # Read query and document files
    queries_df = pd.read_csv(query_file, sep="\t", names=["qid", "query"])
    documents_df = pd.read_csv(document_file, sep="\t", names=["did", "document"])
    
    # Stream triples file in chunks
    for chunk in pd.read_csv(triples_file, sep="\t", names=["qid", "positive_doc_id", "negative_doc_id"], chunksize=batch_size):
        # Merge queries and documents with chunk
        chunk_merged = chunk.merge(queries_df, on="qid")
        chunk_merged = chunk_merged.merge(documents_df, left_on="positive_doc_id", right_on="did", suffixes=("_query", "_positive"))
        chunk_merged = chunk_merged.merge(documents_df, left_on="negative_doc_id", right_on="did", suffixes=("_positive", "_negative"))
        
        # Yield processed examples
        for _, row in chunk_merged.iterrows():
            yield {
                "query": row["query"],
                "positive_document": row["document_positive"],
                "negative_document": row["document_negative"],
            }
config = json.loads(open('config.json').read())

query_file = config['train']['pass']['queries']
document_file = config['train']['pass']['docs']
#triples_file = config['train']['pass']['triples']
triples_file = 'data/msmarco/qidpidtriples.train.full.100k.tsv'


# Create an iterative dataset
dataset = Dataset.from_generator(
    triples_generator)


dataset.save_to_disk("data/msmarco/msmarco.100k.hf")


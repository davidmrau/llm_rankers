import sys
import numpy as np

def read_trec_file(file_path):
    """
    Read a TREC-formatted file and return a dictionary mapping (qid, doc_id) tuples to their scores.
    """
    document_scores = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, doc_id, _, score, _ = parts[:6]
                key = (qid, doc_id)
                document_scores[key] = np.exp(float(score))
    return document_scores

# min_max over all queries
def min_max_normalize(scores):
    """
    Normalize scores between 0 and 1 using min-max normalization.
    """
    min_score = min(scores.values())
    max_score = max(scores.values())
    normalized_scores = {}
    for doc_id, score in scores.items():
        normalized_score = (score - min_score) / (max_score - min_score)
        normalized_scores[doc_id] = normalized_score
    return normalized_scores

def min_max_normalize(scores):
    """
    Normalize scores between 0 and 1 using per-query min-max normalization.
    """
    normalized_scores = {}
    for qid in set(qid for qid, _ in scores.keys()):
        query_scores = [score for (qid_, _), score in scores.items() if qid_ == qid]
        min_score = min(query_scores)
        max_score = max(query_scores)
        for (qid_, doc_id), score in scores.items():
            if qid_ == qid:
                normalized_score = (score - min_score) / (max_score - min_score)
                normalized_scores[(qid_, doc_id)] = normalized_score
    return normalized_scores




def linear_interpolation(scores1, scores2, alpha):
    """
    Perform linear interpolation of two sets of scores with a given interpolation factor alpha.
    """
    interpolated_scores = {}
    for doc_id in scores1:
        score1 = scores1[doc_id]
        score2 = scores2[doc_id]
        interpolated_scores[doc_id] =( (alpha * score1) + ((1 - alpha) * score2))
        #interpolated_scores[doc_id] =(score1 * score2)
    return interpolated_scores



def rank_scores(scores):
    """
    Rank the scores for each query based on the interpolation results.
    """
    ranked_scores = {}
    for (qid, _), score in scores.items():
        if qid not in ranked_scores:
            ranked_scores[qid] = []
        
        # Add unique scores only to the ranked_scores list
        if score not in set(ranked_scores[qid]):
            ranked_scores[qid].append(score)

    # Sort scores in descending order for each query
    for qid in ranked_scores:
        ranked_scores[qid].sort(reverse=True)

    return ranked_scores

def write_trec_file(file_path, interpolated_scores):
    """
    Write the interpolated scores to a new TREC-formatted file with appropriate ranking.
    """
    ranked_scores = rank_scores(interpolated_scores)
    with open(file_path, "w") as outfile:
        for qid, unique_scores in ranked_scores.items():
            rank = 1
            for score in unique_scores:
                for (doc_id, s) in interpolated_scores.items():
                    if qid == doc_id[0] and score == s:
                        outfile.write(f"{qid}\tQ0\t{doc_id[1]}\t{rank}\t{score}\tINTERPOLATED\n")
                        rank += 1
                        break


def get_base(f):
    return f.split('/')[-2]

if __name__ == "__main__":
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    alpha = float(sys.argv[3])
    
    # Read the two TREC-formatted files
    scores1 = read_trec_file(file1_path)
    scores2 = read_trec_file(file2_path)
    
    # Normalize scores between 0 and 1 using min-max normalization
    #normalized_scores1 = min_max_normalize(scores1)
    #normalized_scores2 = min_max_normalize(scores2)
    
    # Perform linear interpolation on normalized scores
    #interpolated_scores = linear_interpolation(normalized_scores1, normalized_scores2, alpha)
    interpolated_scores = linear_interpolation(scores1, scores2, alpha)


    # Write the interpolated scores to a new TREC-formatted file
    write_trec_file(f"interpolated_scores/interpolated_{get_base(file1_path)}_{get_base(file2_path)}_alpha_{alpha}_mult_exp.trec", interpolated_scores)    

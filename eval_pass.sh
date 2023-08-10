#QREL=data/msmarco_docs/2020qrels-docs.txt 
QREL=data/msmarco/2020qrels-pass.txt

MAP=`./trec_eval  $QREL $1 -m all_trec  -l 2| grep -w -E 'map_cut_1000'| awk '{print $3}'`
RECALLH=`./trec_eval $QREL $1 -m all_trec -l 2 | grep -w -E 'recall_100' | awk '{print $3}'`
RECALLTH=`./trec_eval $QREL $1 -m all_trec -l 2 | grep -w -E 'recall_30' | awk '{print $3}'`
RECALLK=`./trec_eval $QREL $1 -m all_trec -l 2 | grep -w -E 'recall_1000' | awk '{print $3}'`


NDCG=`./trec_eval $QREL $1 -m all_trec| grep -w 'ndcg_cut_10' | awk '{print $3}'`
 
deci(){
	printf "%.2f\n" $(echo "scale=0; ${1}*100" | bc)
}

NDCG=`deci $NDCG`
MAP=`deci $MAP`
RECALLH=`deci $RECALLH`
RECALLK=`deci $RECALLK`
RECALLTH=`deci $RECALLTH`
#echo NDCG " & " MAP " & " RECALLTH " & " RECALLH " & " RECALLK \\\\  
#echo $NDCG " & " $MAP " & " $RECALLTH " & " $RECALLH " & " $RECALLK \\\\  
echo " & " $NDCG " & " $MAP  \\\\  





#./trec_eval data/msmarco/2020qrels-pass.txt $1 -m all_trec -q >> ${1}.all_scores.trec

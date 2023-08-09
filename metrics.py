import sys
import tempfile
import subprocess



def write_ranking(scores, q_ids, results_file_path):

	results_file = open(results_file_path, 'w')
	for i, q_id in enumerate(q_ids):
		for j, (doc_id, score) in enumerate(scores[i]):
			results_file.write(f'{q_id}\t{doc_id}\t{j+1}\n' )

	results_file.close()



def write_ranking_trec(scores, q_ids, results_file_path):

	results_file = open(results_file_path, 'w')
	for i, q_id in enumerate(q_ids):
		for j, (doc_id, score) in enumerate(scores[i]):
			results_file.write(f'{q_id}\tQ0\t{doc_id}\t{j+1}\t{score}\teval\n')
	results_file.close()








class TrecEval(object):
	def __init__(self, trec_eval_path):
		self.path = trec_eval_path

	def score(self, qrel_path, ranking_path, max_rank, add_params=''):

		#all topics
		output_all_topics = subprocess.check_output(f"./{self.path} {add_params} -q -m all_trec {qrel_path} {ranking_path} -M {max_rank}", shell=True).decode(sys.stdout.encoding)
		self.ranking_path = ranking_path
		all_topics_path = self.ranking_path + '.scores_all_topics.trec'
		#print(all_topics_path)
		with open(all_topics_path, 'w') as f:
			f.write(output_all_topics)


		# overview trec_eval
		output = subprocess.check_output(f"./{self.path} {add_params} {qrel_path} {ranking_path} -M {max_rank} -m all_trec", shell=True).decode(sys.stdout.encoding)

		with open(self.ranking_path + '.scores.trec', 'w') as f:
			f.write(output)
		return output

class Metric(object):

	def __init__(self, max_rank, qrel_file, ranking_file_path=None):
		self.name = None
		self.max_rank = max_rank
		self.qrel_file = qrel_file
		if ranking_file_path:
			self.ranking_file_path = ranking_file_path
		else:
			tmp = tempfile.NamedTemporaryFile()
			self.ranking_file_path = tmp.name

	def write_scores(self, scores, qids, path):
		raise NotImplementedError()

	def score(self):
		raise NotImplementedError()


class Trec(Metric):
	def __init__(self, metric, trec_eval_path, qrel_file, max_rank, add_params='', ranking_file_path=None):
		super().__init__(max_rank, qrel_file, ranking_file_path)
		self.name = f'{metric}@{max_rank}'
		self.add_params = add_params
		self.trec_eval = TrecEval(trec_eval_path)
		self.metric = metric

	def write_scores(self, scores, qids, path):
		write_ranking(scores, qids, f'{path}.tsv')
		write_ranking_trec(scores, qids, f'{path}.trec')

	def score(self, scores, qids, save_path=''):
		if save_path:
			path = save_path
		else:
			path = self.ranking_file_path

		self.write_scores(scores, qids, path)
		output = self.trec_eval.score(self.qrel_file, f'{path}.trec', self.max_rank, self.add_params)

		output = output.replace('\t', ' ').split('\n')
		for line in output:
			if line.startswith(self.metric):
				return round(float(line.split()[2]), 6)

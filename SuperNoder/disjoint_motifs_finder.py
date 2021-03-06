from  SuperNoder_diff_types.support_functions import Utils
import random
import itertools
import operator
import networkx as nx

class DisjointMotifsFinder:
	def __init__(self, g, motifs, motifs2edges, sample_size, node2motifs, h1_times_repetitions):
		self.g = g
		self.motifs = motifs
		self.disjoint_motifs = {}#set()
		self.motifs2edges = motifs2edges
		self.sample_size = sample_size
		self.node2motifs = node2motifs
		self.h1_times_repetitions = h1_times_repetitions
		
	def h1(self):
		for motif in self.motifs:
			h1_times_repetitions = self.h1_times_repetitions
			self.disjoint_motifs[motif] = dict()
			motifs = (self.motifs[motif])
			while(h1_times_repetitions > 0):
				random.shuffle(motifs)
				seen_nodes = set()
				tmp_disjoint_motifs = set()
				for m in motifs:
					if len(set(m).intersection(seen_nodes)) == 0:
						tmp_disjoint_motifs.add(m)
						seen_nodes = seen_nodes.union(set(m))
				if len(tmp_disjoint_motifs) > len(self.disjoint_motifs[motif]):
					self.disjoint_motifs[motif] = tmp_disjoint_motifs
				h1_times_repetitions -= 1


	def __sampler(self, motifs):
		motifs_list = list(motifs)
		samples = []		
		sample = []
		while len(motifs_list) > 0:
			n = random.randint(0, len(motifs_list) - 1)
			sample += [motifs_list[n]]
			del motifs_list[n]
			if len(sample) == self.sample_size or len(motifs_list) == 0:
				samples += [sample]
				sample = []
		return samples
		
	def __are_there_overlaps(self, l):
		pairs = itertools.combinations(l, 2)
		for pair in pairs:
			if len(set(pair[0]).intersection(set(pair[1]))) > 0:
				#print(set(pair[0]).intersection(set(pair[1])))
				return True
		return False
		
		
	def h2(self):
		for motif in self.motifs:
			self.disjoint_motifs[motif] = dict()
			motifs = self.motifs[motif]
			samples = self.__sampler(motifs)
			print('number of samples:', len(samples), 'of size', self.sample_size)
			while True:
				candidates = set()
				for sample in samples:
					sample_g = nx.Graph()
					enumerated_sample = enumerate(sample)

					for (m1,m2) in itertools.combinations(enumerated_sample, 2):

						sample_g.add_node(m1[0] + 1)
						sample_g.add_node(m2[0] + 1)
						if len(set(m1[1]).intersection(set(m2[1]))) > 0:
							sample_g.add_edge(m1[0] + 1, m2[0] + 1)

					s, l = Utils.clique_removal(sample_g)
					s = [m for i,m in enumerate(sample) if i + 1 in s]
					candidates = candidates.union(s)

				if self.__are_there_overlaps(candidates):
					samples = self.__sampler(list(candidates))
				else:
					self.disjoint_motifs[motif] = candidates
					break

				
	
	
	def h3(self):
		for m in self.motifs:
			motifs_list = self.motifs[m]
			motif_degree_map = {}
			candidates = []

			for motif in motifs_list:
				degree = 0
				for node in motif:
					degree += self.g.degree[node]
				number_of_edges = len(self.motifs2edges[motif])
				degree -= number_of_edges
				motif_degree_map[motif] = degree

			for node in self.node2motifs[m]:
				sub_dict = {k:v for k,v in motif_degree_map.items() if k in self.node2motifs[m][node]}

				sorted_motifs = [x[0] for x in sorted(sub_dict.items(), key=operator.itemgetter(1))]

				if len(sorted_motifs) > 0:
					candidates += [sorted_motifs[0]]

			i = 0
			j = 0
			overlaps = True
			while overlaps:
				overlaps = False
				for i in range(0, len(candidates) - 1):
					for j in range(i + 1, len(candidates)):
						if len(set(candidates[i]).intersection(set(candidates[j]))) != 0:
							overlaps = True
							break
					if overlaps:
						break
				if overlaps:
					mi = candidates[i]
					mj = candidates[j]
					if motif_degree_map[mi] > motif_degree_map[mj]:
						del candidates[i]
					elif motif_degree_map[mi] < motif_degree_map[mj]:
						del candidates[j]
					else:
						h = random.choice([i,j])
						del candidates[h]

			self.disjoint_motifs[m] = set(candidates)

	def h4(self):
		self.h3()
		self.disjoint_motifs_tmp = self.disjoint_motifs
		for motif_name in self.disjoint_motifs_tmp:
			candidates = list(self.disjoint_motifs_tmp[motif_name])
			covered_nodes = set([x for m in candidates for x in m])
			all_nodes = set([x for x in self.g.nodes()])
			orphans = all_nodes - covered_nodes
			for node in orphans:
				if node in self.node2motifs[motif_name]:
					for m in self.node2motifs[motif_name][node]:
						if set(m).intersection(covered_nodes) == set() and set(m).issubset(all_nodes):
							candidates += [m]
							covered_nodes = covered_nodes.union(set(m))
							break
			self.disjoint_motifs[motif_name] = set(candidates)
		
				
	
	def h5(self):
		for motif_name in self.motifs:
			mm = self.motifs[motif_name]
			samples = self.__sampler(mm)
			while True:
				candidates = set()
				for sample in samples:
					sample_g = nx.Graph()
					enumerated_sample = enumerate(sample)

					for (m1,m2) in itertools.combinations(enumerated_sample, 2):
						sample_g.add_node(m1[0] + 1)
						sample_g.add_node(m2[0] + 1)
						if len(set(m1[1]).intersection(set(m2[1]))) > 0:
							sample_g.add_edge(m1[0] + 1, m2[0] + 1)

					d = dict([(n, sample_g.degree[n]) for n in sample_g.nodes()])
					sorted_d = [x[0] for x in sorted(d.items(), key=operator.itemgetter(1))]

					selected = set()
					ignore_list = []
					for n in sorted_d:
						if n not in ignore_list:
							selected.add(n)
							ignore_list += sample_g.neighbors(n)
					s = [m for i,m in enumerate(sample) if i + 1 in selected]
					candidates = candidates.union(s)

				if self.__are_there_overlaps(candidates):
					samples = self.__sampler(list(candidates))
				else:
					self.disjoint_motifs[motif_name] = candidates
					break

			
	
	def run(self, mode):
		if mode == 'h1':
			self.h1()
		elif mode == 'h2':
			self.h2()
		elif mode == 'h3':
			self.h3()
		elif mode == 'h4':
			self.h4()
		elif mode == 'h5':
			self.h5()
			
	def get_disjoint_motifs(self):
		return self.disjoint_motifs
		
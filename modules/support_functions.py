class Utils:
	@staticmethod
	def motifs_to_vec(motifs1, motifs2):
			import numpy as np
			names_of_all_motifs=[] #list of all motif types in all datasets
			for name_of_motif in motifs1:
				if name_of_motif not in names_of_all_motifs:
						names_of_all_motifs.append(str(name_of_motif))
			for name_of_motif in motifs2:
				if name_of_motif not in names_of_all_motifs:
					names_of_all_motifs.append(str(name_of_motif))
			names_of_all_motifs=sorted(names_of_all_motifs)
			new_motifs = np.zeros((2,len(names_of_all_motifs)))
			sum_motif_1 =  0
			sum_motif_2 =  0
			for motif in motifs1:
				sum_motif_1+=motifs1[motif]
			for motif in motifs2:
				sum_motif_2+=motifs2[motif]
			for k,m in enumerate(names_of_all_motifs):
				if m in motifs1:
						new_motifs[0][k] = motifs1[m]/sum_motif_1
				if m in motifs2:
					new_motifs[1][k] = motifs2[m]/sum_motif_2
			return new_motifs[0], new_motifs[1]
	@staticmethod
	def find_motifs_all_types(g,ms_max):
			from SuperNoder.manager import Manager as Manager
            
			arg_tn='undirect'
			arg_th = '1' 
			arg_m = 'h1'
			arg_ss = '100'
			arg_h1tr = 1 
    
			distributions = {}
			distributions_disjoint = {}
    
			for arg_ms in range(3,ms_max+1):
				argv=[' ','-g',g[1],'-th',arg_th,'-ms',arg_ms,'-m',arg_m,\
			'-h1tr',arg_h1tr,'-ss',arg_ss]
				m = Manager(argv)
				distribution_f1 = m.run()
				distribution_disjoint = m.disjoint_finder()
				distributions = dict(list(distributions.items()) + list(distribution_f1.items()))
				distributions_disjoint = dict(list(distributions_disjoint.items()) + list(distribution_disjoint.items()))
			return  g[0],distributions,distributions_disjoint#первое значение словаря - тип мотива или размер мотива, второе значение - количество таких мотивов в графе	
	@staticmethod
	def find_motifs_diff_types(g,ms_max):
		from SuperNoder_diff_types.manager import Manager as Manager_types 
		arg_tn='undirect'
		arg_th = '1' 
		arg_m = 'h1'
		arg_ss = '100'
		arg_h1tr = 1 
    
		distributions = {}
		distributions_disjoint = {} 
		for arg_ms in range(3,ms_max+1):
			argv=[' ','-g',g[1],'-th',arg_th,'-ms',arg_ms,'-m',arg_m,\
      '-h1tr',arg_h1tr,'-ss',arg_ss]
			m = Manager_types(argv)
			distribution_f1 = m.run()
			distribution_disjoint = m.disjoint_finder()
			distributions = dict(list(distributions.items()) + list(distribution_f1.items()))
			distributions_disjoint = dict(list(distributions_disjoint.items()) + list(distribution_disjoint.items()))
		return  g[0],distributions,distributions_disjoint#первое значение словаря - тип мотива или размер мотива, второе значение - количество таких мотивов в графе

	@staticmethod
	def find_motifs(inp,
					ms_max=8):  # возвращает motifs f1 И f3. Разделение на разные типы мотивов. Размеры мотивов 3 и 4
		method, number_of_nodes = inp
		import pickle
		from modules.support_functions import Utils

		motifs_f1 = dict()
		motifs_f3 = dict()
		for graph in graphs:
			if number_of_nodes <= graph[1].number_of_nodes():
				if method == RecursiveModularity:

					sim_tuple = [tuple([k[0], k[1], 1]) for k in graph[1].edges()]
					graph_i = igraph.Graph.TupleList(sim_tuple, weights=True)

					rm = RecursiveModularity(graph_i, min_modularity=0.1, min_nodes=number_of_nodes,
											 modularity_iters=10)
					G_tree = rm.calculate_tree()
					# finding the index of module with the closest number of nodes to the needed one
					sg_names = list(rm.node_popualation.keys())  # all indices of all modules

					me = len(max(list(map(lambda x: x.split('_'), sg_names)), key=len))

					if len(min(list(map(lambda x: x.split('_'), sg_names)), key=len)) != me:

						indices = [i for i, j in enumerate(sg_names) if len(j.split('_')) == me]
						sg_names_max = [sg_names[i] for i in indices]

						def func(x, prev):
							if x:
								return prev + x
							else:
								return prev

						d = []
						candidates = reduce(func, list(
							map(lambda x: re.findall('_'.join(x[1].split('_')[:-6]) + '_' + '\d{1,3}' + '_', x[0]),
								product(sg_names, sg_names_max))), d)
						l_cand_sep = list(Counter(candidates).keys())

					else:
						l_cand_sep = sg_names

					min_ = graph_i.vcount()
					min_ind = l_cand_sep[0]
					for i in (l_cand_sep):
						if len(rm.node_popualation[i]) > number_of_nodes and len(rm.node_popualation[i]) < min_:
							min_ = len(rm.node_popualation[i])
							min_ind = i

					# finding the sample
					sample = graph_i.subgraph([x.index for x in graph_i.vs if x['name'] in rm.node_popualation[
						min_ind]])  # should write another index
					sample = sample.to_networkx()

					for node in sample.nodes:
						sample.add_node(node, label='Motif')
					_, motifs_sample, motifs_disjoint_sample = Utils.find_motifs_diff_types((graph[0], sample), ms_max)

					motifs_f1[graph[0] + '_0'] = motifs_sample
					motifs_f3[graph[0] + '_0'] = motifs_disjoint_sample

				else:  # for other methods we should
					for s in range(10, 20):  # add repeat due to stochastic methods
						sampler = method(number_of_nodes, seed=s)
						sample = sampler.sample(graph[1])
						if not sample.nodes[list(sample.nodes)[0]]:  # если первая вершина в списке нулевая.
							for node in sample.nodes:
								sample.add_node(node, label='Motif')

						_, motifs_sample, motifs_disjoint_sample = Utils.find_motifs_diff_types((graph[0], sample),
																								ms_max)

						motifs_f1[graph[0] + '_' + str(s - 10)] = motifs_sample
						motifs_f3[graph[0] + '_' + str(s - 10)] = motifs_disjoint_sample
			else:
				for s in range(10):
					motifs_f1[graph[0] + '_' + str(s)] = {}
					motifs_f3[graph[0] + '_' + str(s)] = {}
		return number_of_nodes, motifs_f1, motifs_f3
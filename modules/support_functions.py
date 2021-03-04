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
	def find_motifs(g):
			from SuperNoder.manager import Manager as Manager
            
			arg_tn='undirect'
			arg_th = '1' 
			arg_m = 'h1'
			arg_ss = '100'
			arg_h1tr = 1 
    
			distributions = {}
			distributions_disjoint = {}
    
			for arg_ms in range(3,5):
				argv=[' ','-g',g[1],'-th',arg_th,'-ms',arg_ms,'-m',arg_m,\
			'-h1tr',arg_h1tr,'-ss',arg_ss]
				m = Manager(argv)
				distribution_f1 = m.run()
				distribution_disjoint = m.disjoint_finder()
				distributions = dict(list(distributions.items()) + list(distribution_f1.items()))
				distributions_disjoint = dict(list(distributions_disjoint.items()) + list(distribution_disjoint.items()))
			return  g[0],distributions,distributions_disjoint#первое значение словаря - тип мотива или размер мотива, второе значение - количество таких мотивов в графе	
	@staticmethod
	def find_motifs_diff_types(g):
		from SuperNoder_diff_types.manager import Manager as Manager_types 
		arg_tn='undirect'
		arg_th = '1' 
		arg_m = 'h1'
		arg_ss = '100'
		arg_h1tr = 1 
    
		distributions = {}
		distributions_disjoint = {}
    
		for arg_ms in range(3,4):
			argv=[' ','-g',g[1],'-th',arg_th,'-ms',arg_ms,'-m',arg_m,\
      '-h1tr',arg_h1tr,'-ss',arg_ss]
			m = Manager_types(argv)
			distribution_f1 = m.run()
			distribution_disjoint = m.disjoint_finder()
			distributions = dict(list(distributions.items()) + list(distribution_f1.items()))
			distributions_disjoint = dict(list(distributions_disjoint.items()) + list(distribution_disjoint.items()))
		return  g[0],distributions,distributions_disjoint#первое значение словаря - тип мотива или размер мотива, второе значение - количество таких мотивов в графе
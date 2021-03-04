import sys
import os
import networkx as nx
import itertools
import traceback

#supernoder imports
from SuperNoder.enumerator import *
from SuperNoder.disjoint_motifs_finder import *
from SuperNoder.counter import *

class Manager:
	TOOL_NAME = 'SUPERNODER 1.0'
	
	def __init__(self, argv):
		#general params
		self.nodes_string = ''
		self.edges_string = ''
		self.web_input = False
		self.argv = argv
		self.g = None
		self.reduced_g = None
		self.edges_file = None
		self.nodes_file = None
		self.type_of_network = 'direct'
		self.method = 'h1'
		self.motifs = set()
		self.motif2edges = {}
		self.motifs_size = 3
		self.th = 50
		self.n_levels = 1
		self.h1_times_repetitions = 1
		self.sample_size = 100
		self.supernoder_out_name = 'result/supernoder_output'
		self.node2motifs = {}
		self.disjoint_motifs = set()

		#components
		self.enumerator = None
		self.disjoint_motif_finder = None
		self.counter = None
		
		
	def __manage_input(self):
		if len(self.argv) <= 1:
			print("No param has been provided. Please see: python supernoder.py --help")
			sys.exit(1)
		else:
			i = 1
			while i < len(self.argv):
				if self.argv[i] == '--graph' or self.argv[i] == '-g':
					self.g=self.argv[i+1]
					i += 2
					self.type_of_network = 'direct' if nx.is_directed(self.g) else 'undirect'
				elif self.argv[i] == '--threshold' or self.argv[i] == '-th':
					try:
						self.th = int(self.argv[i + 1])
						i += 2
					except:
						print('ERROR <threshold> must be a number')
						sys.exit(1)
				elif self.argv[i] == '--motif-size' or self.argv[i] == '-ms':
					try:
						self.motifs_size = int(self.argv[i + 1])
						i += 2
					except:
						print('ERROR <motif_size> must be a number')
						sys.exit(1)
				elif self.argv[i] == '--method' or self.argv[i] == '-m':
					if self.argv[i + 1] in ['h1', 'h2', 'h3', 'h4', 'h5']:
						self.method = self.argv[i + 1]
						i += 2
					else:
						print('ERROR Chosen method is not recognized')
						sys.exit(1)
				
				elif self.argv[i] == '--h1-times-repetition' or self.argv[i] == '-h1tr':
					try:
						self.h1_times_repetitions = int(self.argv[i + 1])
						i += 2
					except:
						print('ERROR <times_repetitions> must be a number')
						sys.exit(1)
				elif self.argv[i] == '--samples-size' or self.argv[i] == '-ss':
					try:
						self.sample_size = int(self.argv[i + 1])
						i += 2
					except:
						print('ERROR <sample_size> must be a number')
						sys.exit(1)
				elif self.argv[i] == '--web-input' or self.argv[i] == '-w':
					try:
						if self.argv[i + 1] == '1' or self.argv[i + 1] == 'True':
							self.web_input = True
							i += 2
						else:
							print('WEB mode has not been recognized')
							sys.exit(1)
					except:
						sys.exit(1)
						
				elif self.argv[i] == '--help' or self.argv[i] == '-h':
					print("\n#####################################\nRelease 1.0 SUPERNODER\n")
					print("Usage: python xproject -t <input_text> \n LIST OF PARAMS:\n" \
					" -g,  --graphs \t\t<filename> \tMANDATORY \tthe networkx graph\n" \
					" -th, --threshold \t\t<threshold> \tOPTIONAL \tThe threshold to hold over-represented motifs.\n" \
					" -ms, --motif-size \t\t<size> \t\tOPTIONAL \tThe size of motifs. It must be greater or equal to 3. DEFAULT: 3.\n"\
					" -m,  --method \t\t\t<method> \tOPTIONAL \tThe heuristic to use in order to maximize motifs. DEFAULT: 3\n" \
					" -h1tr, --h1-times-repetition \t<times> \tOPTIONAL \tThe number of repetition of h1. DEFAULT: 1.\n" \
					" -ss, --samples-size \t\t<sample_size> \tOPTIONAL \tThe size of samples for heuristics h4 and h5. DEFAULT: 100.\n")
					sys.exit(1)
				else:
					print("\nERROR Param: # " + str(self.argv[i]) + " # Please see: python supernoder.py --help")
					sys.exit(1)
		
	
	def __load_graph(self):	
			pass#print ('#\tThe original network has ', len(self.g.nodes()), ' nodes and ', len(self.g.edges()), ' edges')
	def __compile_node2motifs(self):
		self.node2motifs = {}   
		for motif_name in self.motifs:
			self.node2motifs[motif_name] ={}
			for motif in self.motifs[motif_name]:
				for n in motif:
					if n not in self.node2motifs[motif_name]:
						self.node2motifs[motif_name][n] = set()
					self.node2motifs[motif_name][n].add(motif)            
			#for n in motif:
			#	if n not in self.node2motifs:
			#		self.node2motifs[n] = set()
			#	self.node2motifs[n].add(motif)
		
	def	__do_enumeration(self):
		self.motifs = Utils.enumerate_motifs(self.g, self.motifs_size) 
		self.motif2edges = {}
		for motif in self.motifs:
			if motif not in self.motif2edges:
				self.motif2edges[motif] = set()
			for v in itertools.combinations(motif, 2):
				if self.type_of_network == 'direct':
					if self.g.has_edge(v[0], v[1]):
						self.motif2edges[motif].add(v)
					if self.g.has_edge(v[1], v[0]):
						self.motif2edges[motif].add((v[1],v[0]))
				elif self.type_of_network == 'undirect':
					if self.g.has_edge(v[0], v[1]):
						self.motif2edges[motif].add(v)
                        
	def __find_disjoint_motifs(self): 
		self.disjoint_motif_finder = DisjointMotifsFinder(self.g, self.motifs, self.motif2edges, self.sample_size, self.node2motifs, self.h1_times_repetitions)
		self.disjoint_motif_finder.run(self.method)
		self.disjoint_motifs = self.disjoint_motif_finder.get_disjoint_motifs()
		#print('#\tThe number of disjoint motifs is ',len([x for k,v in self.disjoint_motifs.items() for x in v]))

	def __cut(self):
		self.counter = Counter(self.g, self.motifs, self.th, self.type_of_network, self.motif2edges)
		self.counter.run() 
		self.motifs = self.counter.get_selected_motifs()
		#print ('#\tThe number of motifs that occur more than', self.th, 'times is',len([x for k,v in self.motifs.items() for x in v]))
        
	def run(self):
		self.__manage_input()
		self.__load_graph()
		self.__do_enumeration()     
		self.__cut()    
		self.__compile_node2motifs()
		distribution_f1 = dict()
		for motif_name in self.motifs:     
			distribution_f1[motif_name] = len(self.motifs[motif_name])
		return distribution_f1
	def disjoint_finder(self):
		self.__find_disjoint_motifs()        
		distribution_disjoint =dict()
		for motif_name in self.disjoint_motifs:
			distribution_disjoint[motif_name]=len(self.disjoint_motifs[motif_name])
		return distribution_disjoint
import sys
import os
import networkx as nx
import itertools
import traceback

#supernoder imports
from SuperNoder.enumerator import *
from SuperNoder.disjoint_motifs_finder import *
from SuperNoder.counter import *
import multiprocessing as mp
p = mp.Pool(mp.cpu_count())
parellel = True # change to True to compute in parallel
chunksize = 25 # vary this value to speedup parallel computation

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
		print('\n' + self.TOOL_NAME + ' Reading input')
		if len(self.argv) <= 1:
			print("No param has been provided. Please see: python supernoder.py --help")
			sys.exit(1)
		else:
			i = 1
			while i < len(self.argv):
				if self.argv[i] == '--edges-file' or self.argv[i] == '-e':
					if os.path.exists(self.argv[i + 1]):
						self.edges_file = self.argv[i + 1]
						i += 2
					else:
						print('ERROR The file ' + self.argv[i + 1] + ' does not exist')
						sys.exit(1)
						
				elif self.argv[i] == '--nodes-file' or self.argv[i] == '-n':
					if os.path.exists(self.argv[i + 1]):
						self.nodes_file = self.argv[i + 1]
						i += 2
					else:
						print('ERROR The file ' + self.argv[i + 1] + ' does not exist')
						sys.exit(1)
						
				elif self.argv[i] == '--type-of-network' or self.argv[i] == '-tn':
					if self.argv[i + 1] in ['direct', 'undirect']:
						self.type_of_network = self.argv[i + 1]
						i += 2
					else:
						print("ERROR <type>: # " + str(self.argv[i + 1]) + " # Network can only be direct or indirect Please see: python supernoder.py --help")		
						sys.exit(1)
					
				
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
					" -n,  --nodes-file \t\t<filename> \tMANDATORY \tThe list of nodes. Node id and label for each row separated by a space\n" \
					" -e,  --edges-file \t\t<filename> \tMANDATORY \tThe list of edges. One edge for each row.\n" \
					" -m,  --method \t\t\t<method> \tOPTIONAL \tThe heuristic to use in order to maximize motifs. DEFAULT: 3\n" \
					" -tn, --type-of-network \t<type> \t\tOPTIONAL 	The type of network. It can be chosen from [direct, undirect]. DEFAULT: direct.\n" \
					" -th, --threshold \t\t<threshold> \tOPTIONAL \tThe threshold to hold over-represented motifs.\n" \
					" -ms, --motif-size \t\t<size> \t\tOPTIONAL \tThe size of motifs. It must be greater or equal to 3. DEFAULT: 3.\n"\
					" -h1tr, --h1-times-repetition \t<times> \tOPTIONAL \tThe number of repetition of h1. DEFAULT: 1.\n" \
					" -ss, --samples-size \t\t<sample_size> \tOPTIONAL \tThe size of samples for heuristics h4 and h5. DEFAULT: 100.\n")
					sys.exit(1)
				else:
					print("\nERROR Param: # " + str(self.argv[i]) + " # Please see: python supernoder.py --help")
					sys.exit(1)
		
	
	def __load_graph(self):	
		print ('# ' + (str(datetime.datetime.now()) + ' Loading graph'))
		if self.type_of_network == 'direct':
			self.g = nx.DiGraph()	
		elif self.type_of_network == 'undirect':
			self.g = nx.Graph()	
		try:							
			with open(self.nodes_file, 'r') as f:
				lines = f.read().splitlines()
				for line in lines:
					values = line.split(' ')
					if len(values) == 2:
						self.g.add_node(values[0], label=values[1])	
					elif len(values) > 2 or len(values) == 1:
						print('nodes are not well formatted')
						raise ValueError('nodes are not well formatted') 
			nodes_set = set(self.g.nodes())
			with open(self.edges_file, 'r') as f:
				lines = f.read().splitlines()
				for line in lines:
					values = line.split(' ')
					if len(values) == 2 and values[0] in nodes_set and values[1] in nodes_set:
						self.g.add_edge(values[0], values[1])
					elif len(values) > 2 or len(values) == 1:
						print('edges are not well formatted')
						raise ValueError('edges are not well formatted') 
			
			print ('#\tThe original network has ', len(self.g.edges()), ' nodes and ', len(self.g.edges()), ' edges')
		except:
			traceback.print_exc(file=sys.stdout)
			print("ERROR: nodes file or edges file is missing")
			sys.exit(1)
					
	def __compile_node2motifs(self):#NO NEED IN PARALLEL
		self.node2motifs = {}
		for motif in self.motifs:
			for n in motif:
				if n not in self.node2motifs:
					self.node2motifs[n] = set()
				self.node2motifs[n].add(motif)

	def	__do_enumeration(self):
		print('# ' + str(datetime.datetime.now()) + ' Enumeration')
		self.motifs = Utils.enumerate_motifs(self.g, self.motifs_size) #Utils.enumerate_motifs NeedInParallel
		print ('#\tThe total number of motifs is ', len(self.motifs))
		self.motif2edges = {}
		for motif in motifs:
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
	def __find_disjoint_motifs(self): #Seems like NO NEED IN PARALLEN но надо проверить
		print ('# ' + str(datetime.datetime.now()) + ' Disjointness computation' )
		self.disjoint_motif_finder = DisjointMotifsFinder(self.g, self.motifs, self.motif2edges, self.sample_size, self.node2motifs, self.h1_times_repetitions)
		print( 'Ниже построение run' + str(datetime.datetime.now()))
		self.disjoint_motif_finder.run(self.method)
		print( 'Ниже построение get_disjoint_motifs' + str(datetime.datetime.now()))
		self.disjoint_motifs = self.disjoint_motif_finder.get_disjoint_motifs()
		print ('#\tThe number of disjoint motifs is ', len(self.disjoint_motifs))

	def __cut(self):
		print ('# ' + str(datetime.datetime.now()) + ' Threshold computation')
		self.counter = Counter(self.g, self.motifs, self.th, self.type_of_network, self.motif2edges)
		self.counter.run() #Need TO Parallel
		self.motifs = self.counter.get_selected_motifs() #NO NEED In PArallel
		print ('#\tThe number of motifs that occur more than', self.th, 'times is', len(self.motifs))
		
		
	def run(self):
		self.__manage_input()
		self.__load_graph()
		self.__do_enumeration()
		self.__cut()
		self.__compile_node2motifs()
		self.__find_disjoint_motifs()
		distribution_disjoint = dict()
		distribution_f1 = dict()     
		distribution_disjoint[self.motifs_size] =len(self.disjoint_motifs)
		distribution_f1[self.motifs_size] = len(self.motifs)
		print ('SuperNoder execution has finished\n')
		return distribution_disjoint, distribution_f1

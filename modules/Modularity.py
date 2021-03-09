import igraph
import leidenalg as la
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class RecursiveModularity:
    def __init__(self, G, include_leafs=False, min_nodes=50, min_modularity=.5, modularity_iters=100):
        self.target_graph = G
        self.result = []
        # параметры
        self.include_leafs = include_leafs
        self.min_nodes = min_nodes
        self.min_modularity = min_modularity
        self.modulairty_iters = modularity_iters
        # мета для анализа
        self.max_depth = 0
        self.modularity_mapping = {}
        self.size_mapping = {}
        self.depth_mapping = {}
        self.node_popualation = {}
        self.tree = igraph.Graph()
        self.clustered_graph = igraph.Graph()
        self.result = []
        
    def get_module(self, G):
        partition = la.find_partition(G, la.ModularityVertexPartition, n_iterations=self.modulairty_iters)
        return partition.modularity, partition.subgraphs()
    
    def get_tree(self, new_graphs, depth, parent):
        depth+=1
        
        if depth > self.max_depth:
            self.max_depth = depth
            
        for n, G1 in enumerate(new_graphs):
            m, graphs = self.get_module(G1)
            
            current = parent+'D_'+str(depth)+'_n_'+str(n)+"_"
            self.modularity_mapping[current] = m
            self.size_mapping[current] = len(G1.vs())
            self.depth_mapping[current] = depth
            self.node_popualation[current] = set([x['name'] for x in G1.vs()])
            
            # критерий остановки, либо число узлов в подграфе меньше порога, либо модулярность меньше порога
            if len(G1.vs) > self.min_nodes and m > self.min_modularity:
                self.result.append(tuple([parent,current]))
                self.get_tree(graphs, depth, current)
            else:
                self.result.append(tuple([parent,current]))
                
                # для интерпретации узлов в рекурсивных модулях
                if self.include_leafs:
                    [self.result.append(tuple([current,x['name']])) for x in G1.vs]
                    
     
    def calculate_tree(self):
        current = 'D0_'
        v = 1
        m, new_graphs = self.get_module(self.target_graph)
        self.modularity_mapping[current] = m
        self.size_mapping[current] = len(self.target_graph.vs())
        self.depth_mapping[current] = self.max_depth
    
        self.get_tree(new_graphs, self.max_depth, current)
        
        # перевод результата в igraph 
        sim_tuple = [tuple([k[0],k[1],v]) if v > 0 else tuple([k[0],k[1],0]) for k in self.result]
        G_tree = igraph.Graph.TupleList(sim_tuple, weights=True)
        
       # print('Max depth', self.max_depth)
       # print(igraph.summary(G_tree))
        self.tree = G_tree
        return G_tree
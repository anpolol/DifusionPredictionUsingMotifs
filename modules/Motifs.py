import os
import pandas as pd
from collections import Counter
import numpy as np
import random
import json
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_squared_error
import igraph
from functools import reduce
from itertools import product
import regex as re
from sklearn.metrics import mean_squared_error
from datetime import datetime

from modules.support_functions import Utils
from modules.Modularity import RecursiveModularity

from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor
# supernoder files:

# вариант, который считает РАЗНЫЕ типы мотивов одного и того же размера
from SuperNoder_diff_types.manager import Manager as Manager_types

# вариант, который считает все мотивы одного размера вместе
from SuperNoder.manager import Manager as Manager

from modules.support_functions import Utils
import pickle
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from littleballoffur import DegreeBasedSampler, \
    PageRankBasedSampler, \
    RandomEdgeSampler, \
    SnowBallSampler, \
    ForestFireSampler, \
    CommunityStructureExpansionSampler, \
    ShortestPathSampler, \
    RandomWalkSampler, \
    RandomWalkWithJumpSampler, \
    MetropolisHastingsRandomWalkSampler, \
    NonBackTrackingRandomWalkSampler, \
    CirculatedNeighborsRandomWalkSampler, \
    CommonNeighborAwareRandomWalkSampler, \
    LoopErasedRandomWalkSampler

methods = [
    # random node sampling
    DegreeBasedSampler,
    PageRankBasedSampler,

    # Random Edge Sampling
    RandomEdgeSampler,
    SnowBallSampler,
    CommunityStructureExpansionSampler,
    ShortestPathSampler,
    # Random-Walks Dased
    RandomWalkSampler,
    RandomWalkWithJumpSampler,
    MetropolisHastingsRandomWalkSampler,
    NonBackTrackingRandomWalkSampler,
    CirculatedNeighborsRandomWalkSampler,
    CommonNeighborAwareRandomWalkSampler,
    LoopErasedRandomWalkSampler,
    RecursiveModularity
]


def find_motifs(inp,graphs, ms_max=8, #Только для сэмплирования
                diff_types=True):  # возвращает motifs f1 И f3. Разделение на разные типы мотивов. Размеры мотивов 3 и 4
    find_motif = Utils.find_motifs_diff_types if diff_types else Utils.find_motifs_all_types
    method, number_of_nodes = inp

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
                sample = graph_i.subgraph([x.index for x in graph_i.vs if
                                           x['name'] in rm.node_popualation[min_ind]])  # should write another index
                sample = sample.to_networkx()

                for node in sample.nodes:
                    sample.add_node(node, label='Motif')
                name = graph[0]+'_'+str(method).split('.')[-1].split("'")[0]+'_'+str(number_of_nodes)
                _, motifs_sample, motifs_disjoint_sample = find_motif((name, sample), ms_max)

                motifs_f1[graph[0] + '_0'] = motifs_sample
                motifs_f3[graph[0] + '_0'] = motifs_disjoint_sample

            else:  # for other methods we should
                for s in range(10, 20):  # add repeat due to stochastic methods
                    sampler = method(number_of_nodes, seed=s)
                    sample = sampler.sample(graph[1])
                    if not sample.nodes[list(sample.nodes)[0]]:  # если первая вершина в списке нулевая.
                        for node in sample.nodes:
                            sample.add_node(node, label='Motif')
                    name = graph[0] + '_' + str(method).split('.')[-1].split("'")[0]+'_'+str(number_of_nodes)
                    _, motifs_sample, motifs_disjoint_sample = find_motif((name, sample),
                                                                          ms_max)

                    motifs_f1[graph[0] + '_' + str(s - 10)] = motifs_sample
                    motifs_f3[graph[0] + '_' + str(s - 10)] = motifs_disjoint_sample
        else:
            for s in range(10):
                motifs_f1[graph[0] + '_' + str(s)] = {}
                motifs_f3[graph[0] + '_' + str(s)] = {}
    return number_of_nodes, motifs_f1, motifs_f3

def find_motifs_method(methods, diff_types,graphs,ms_max):
        for method in methods:
            motifs_methods_f1 = dict()
            motifs_methods_f3 = dict()
            name_of_method = str(method).split('.')[-1].split("'")[0]
            d = datetime.now()
            motifs_methods_f1.setdefault(name_of_method, dict())
            motifs_methods_f3.setdefault(name_of_method, dict())
            r = 300
            l = 10
            step = 10
            # here is a parallelization
            inp = list(zip([method] * int((r - l) / step), list(range(l, r, step))))
            with ThreadPoolExecutor(max_workers=4) as executor:
                res = executor.map(lambda x: find_motifs(x, graphs, ms_max, diff_types), inp)

            for number_of_nodes, motifs_f1, motifs_f3 in res:
                motifs_methods_f1[name_of_method]['Number of nodes: ' + str(number_of_nodes)] = dict()
                motifs_methods_f3[name_of_method]['Number of nodes: ' + str(number_of_nodes)] = dict()
                motifs_methods_f1[name_of_method]['Number of nodes: ' + str(number_of_nodes)] = dict(
                    list(motifs_methods_f1[name_of_method]['Number of nodes: ' + str(number_of_nodes)].items()) + list(
                        motifs_f1.items()))
                motifs_methods_f3[name_of_method]['Number of nodes: ' + str(number_of_nodes)] = dict(
                    list(motifs_methods_f3[name_of_method]['Number of nodes: ' + str(number_of_nodes)].items()) + list(
                        motifs_f3.items()))
            print(datetime.now() - d)
        return motifs_methods_f1, motifs_methods_f3


def Motifs(diff_types,graphs,ms_max):
    l = 10
    r = 300
    step = 10
    if diff_types:
        print('Counting motifs of different types of initial graphs')
    else:
        print('Counting motifs (without differentiation into types) of initial graphs')

    find_motif = Utils.find_motifs_diff_types if diff_types else Utils.find_motifs_all_types
    # подсчитаем распределение мотивов для исходных графов - надо для подсчета MSE

    d = datetime.now()

    if diff_types:
        if os.path.exists('./DataHelp/motifs_of_full_graphs_f3_diff.pickle') and os.path.exists('./DataHelp/motifs_of_full_graphs_f1_diff.pickle'):
            with open('./DataHelp/motifs_of_full_graphs_f3_diff.pickle','rb') as f:
                motifs_full_graphs_f3_diff = pickle.load(f)
            with open('./DataHelp/motifs_of_full_graphs_f1_diff.pickle','rb') as f:
                motifs_full_graphs_f1_diff = pickle.load(f)
        else:
            with ThreadPoolExecutor(max_workers=4) as executor:
                res = executor.map(lambda x: find_motif(x, ms_max), graphs)
            motifs_full_graphs_f1_diff = dict()
            motifs_full_graphs_f3_diff = dict()
            for name, motifs,motifs_disjoint in res:
                motifs_full_graphs_f1_diff[name] = motifs
                motifs_full_graphs_f3_diff[name] = motifs_disjoint
                with open('./DataHelp/motifs_of_full_graphs_f1_diff.pickle','wb') as f:
                    pickle.dump(motifs_full_graphs_f1_diff,f)
                with open('./DataHelp/motifs_of_full_graphs_f3_diff.pickle','wb') as f:
                    pickle.dump(motifs_full_graphs_f3_diff,f)
        print(datetime.now() - d)

    else:
        if os.path.exists('./DataHelp/motifs_of_full_graphs_f3.pickle') and os.path.exists('./DataHelp/motifs_of_full_graphs_f1_diff.pickle'):
            with open('./DataHelp/motifs_of_full_graphs_f3.pickle','rb') as f:
                motifs_full_graphs_f3 = pickle.load(f)
            with open('./DataHelp/motifs_of_full_graphs_f1.pickle','rb') as f:
                motifs_full_graphs_f1 = pickle.load(f)
        else:
            with ThreadPoolExecutor(max_workers=4) as executor:
                res = executor.map(lambda x: find_motif(x, ms_max), graphs)
            motifs_full_graphs_f1 = dict()
            motifs_full_graphs_f3 = dict()
            for name, motifs,motifs_disjoint in res:
                motifs_full_graphs_f1[name] = motifs
                motifs_full_graphs_f3[name] = motifs_disjoint
                with open('./DataHelp/motifs_of_full_graphs_f1.pickle','wb') as f:
                    pickle.dump(motifs_full_graphs_f1,f)
                with open('./DataHelp/motifs_of_full_graphs_f3.pickle','wb') as f:
                    pickle.dump(motifs_full_graphs_f3,f)
        print(datetime.now() - d)

    print('counting motifs for samples')
    # получим словарь распределения мотивов для сэмплированных подграфов

    if not diff_types:
        if os.path.exists('./motifs_methods_f1.pickle') and os.path.exists('./motifs_methods_f3.pickle'):
            with open('./motifs_methods_f1.pickle') as f:
                motifs_methods_init = pickle.load(f)
                motifs_methods_f1 = dict()
                for method in motifs_methods_init:
                    motifs_methods_f1[method] = dict()
                    for nn in motifs_methods_init[method]:
                        motifs_methods_f1[method][nn] = dict()
                        for dataset in motifs_methods_init[method][nn]:
                            motifs_methods_f1[method][nn][dataset] = dict()
                            motifs_methods_f1[method][nn][dataset] = dict(filter(lambda x: int(x[0].split('_')[1]) <= ms_max,motifs_methods_init[method][nn][dataset].items()))

            with open('./motifs_methods_f3.pickle') as f:
                motifs_methods_init = pickle.load(f)
                motifs_methods_f3 = dict()
                for method in motifs_methods_init:
                    motifs_methods_f3[method] = dict()
                    for nn in motifs_methods_init[method]:
                        motifs_methods_f3[method][nn] = dict()
                        for dataset in motifs_methods_init[method][nn]:
                            motifs_methods_f3[method][nn][dataset] = dict()
                            motifs_methods_f3[method][nn][dataset] = dict(
                                filter(lambda x: int(x[0].split('_')[1]) <= ms_max,
                                       motifs_methods_init[method][nn][dataset].items()))
        else:
            motifs_methods_f1,motifs_methods_f3=find_motifs_method(methods,False,graphs,ms_max)
    else:
        motifs_methods_f1, motifs_methods_f3 = find_motifs_method(methods, True,graphs,ms_max)

    # загрузка мотивов для полных графов, на случай чтоб не пересчитывать

    if diff_types:
        names_of_all_motifs_diff=[]
        for dataset in motifs_full_graphs_f1_diff:
            for name_of_motif in motifs_full_graphs_f1_diff[dataset]:
                if name_of_motif not in names_of_all_motifs_diff:
                    names_of_all_motifs_diff.append(str(name_of_motif))
        for method in motifs_methods_f1:
            for nn in motifs_methods_f1[method]:
                for dataset in motifs_methods_f1[method][nn]:
                    for name_of_motif in motifs_methods_f1[method][nn][dataset]:
                        if name_of_motif not in names_of_all_motifs_diff:
                            names_of_all_motifs_diff.append(str(name_of_motif))
        names_of_all_motifs_diff=sorted(names_of_all_motifs_diff)
        with open('./DataHelp/names_of_all_motifs_diff.pickle','wb') as f:
             pickle.dump(names_of_all_motifs_diff, f)
    else:
        names_of_all_motifs=[]
        for dataset in motifs_full_graphs_f1:
            for name_of_motif in motifs_full_graphs_f1[dataset]:
                if name_of_motif not in names_of_all_motifs:
                    names_of_all_motifs.append(str(name_of_motif))
        for method in motifs_methods_f1:
            for nn in motifs_methods_f1[method]:
                for dataset in motifs_methods_f1[method][nn]:
                    for name_of_motif in motifs_methods_f1[method][nn][dataset]:
                        if name_of_motif not in names_of_all_motifs:
                            names_of_all_motifs.append(str(name_of_motif))
        names_of_all_motifs=sorted(names_of_all_motifs)
        with open('./DataHelp/names_of_all_motifs.pickle','wb') as f:
             pickle.dump(names_of_all_motifs, f)

    # Составляем матрицы частот мотивов для каждого исходного графа
    if diff_types:
        # Матрица распредления f1 частот по датасетам в исходном графе
        X_full_f1 = np.zeros((len(graphs), len(names_of_all_motifs_diff)))  # один ряд - один граф

        # Матрица распредления f3 частот по датасетам в исходном графе
        X_full_f3 = np.zeros((len(graphs), len(names_of_all_motifs_diff)))

        index_list = list(range(3, ms_max + 1))
        for i, (ds_name, gr) in enumerate(graphs):
            my_dict_f1 = motifs_full_graphs_f1_diff[ds_name]
            my_dict_f3 = motifs_full_graphs_f3_diff[ds_name]
            sum_diсt_f1 = dict(
                map(lambda i: (i, sum(map(lambda e: (e[1]), filter(lambda e: str(e[0][6]) == str(i), my_dict_f1.items())))),
                    index_list))
            sum_diсt_f3 = dict(
                map(lambda i: (i, sum(map(lambda e: (e[1]), filter(lambda e: str(e[0][6]) == str(i), my_dict_f3.items())))),
                    index_list))

            X_full_f1[i] = list(
                map(lambda x: my_dict_f1[x] / sum_diсt_f1[int(x[6])] if x in my_dict_f1 else 0, names_of_all_motifs_diff))
            X_full_f3[i] = list(
                map(lambda x: my_dict_f3[x] / sum_diсt_f3[int(x[6])] if x in my_dict_f3 else 0, names_of_all_motifs_diff))

        with open('./DataHelp/motifs_matrix_full_f1.npy', 'wb') as f:
            np.save(f, X_full_f1)
        with open('./DataHelp/motifs_matrix_full_f3.npy', 'wb') as f:
            np.save(f, X_full_f3)
    else:
        X_full = np.zeros((len(graphs), len(names_of_all_motifs)))
        index_list = list(range(3, ms_max + 1))
        for i, (ds_name, gr) in enumerate(graphs):
            my_dict_f1 = motifs_full_graphs_f1[ds_name]
            my_dict_f3 = motifs_full_graphs_f3[ds_name]
            X_full[i] = list(map(lambda x: my_dict_f3[x] / my_dict_f1[x] if x in my_dict_f3 else 0, names_of_all_motifs))

        with open('./DataHelp/motifs_matrix_full.npy', 'wb') as f:
            np.save(f, X_full)

    # Составляем матрицы частот мотивов для каждого размера сэмпла и каждого метода:
    if diff_types:
        X_sample_f3 = dict()
        X_sample_f1 = dict()

        for method in methods:
            s_max = 1 if method == RecursiveModularity else 10

            name_of_method = str(method).split('.')[-1].split("'")[0]

            X_sample_f3[name_of_method] = dict()
            X_sample_f1[name_of_method] = dict()
            for number_of_nodes in list(range(l, r, step)):
                X_sample_f3[name_of_method]['Number of nodes: ' + str(number_of_nodes)] = np.zeros(
                    (len(graphs) * s_max, len(names_of_all_motifs_diff)))
                X_sample_f1[name_of_method]['Number of nodes: ' + str(number_of_nodes)] = np.zeros(
                    (len(graphs) * s_max, len(names_of_all_motifs_diff)))

                for s in range(s_max):
                    for i, (ds_name, gr) in enumerate(graphs):
                        my_dict_f1 = motifs_methods_f1[name_of_method]['Number of nodes: ' + str(number_of_nodes)][
                            ds_name + '_' + str(s)]
                        my_dict_f3 = motifs_methods_f3[name_of_method]['Number of nodes: ' + str(number_of_nodes)][
                            ds_name + '_' + str(s)]
                        sum_diсt_f1 = dict(map(lambda i: (
                        i, sum(map(lambda e: (e[1]), filter(lambda e: str(e[0][6]) == str(i), my_dict_f1.items())))),
                                               index_list))
                        sum_diсt_f3 = dict(map(lambda i: (
                        i, sum(map(lambda e: (e[1]), filter(lambda e: str(e[0][6]) == str(i), my_dict_f3.items())))),
                                               index_list))
                        X_sample_f3[name_of_method]['Number of nodes: ' + str(number_of_nodes)][int(i+s*len(graphs))] = list(
                            map(lambda x: my_dict_f3[x] / sum_diсt_f3[int(x[6])] if x in my_dict_f3 else 0,
                                names_of_all_motifs_diff))
                        X_sample_f1[name_of_method]['Number of nodes: ' + str(number_of_nodes)][int(i+s*len(graphs))] = list(
                            map(lambda x: my_dict_f1[x] / sum_diсt_f1[int(x[6])] if x in my_dict_f1 else 0,
                                names_of_all_motifs_diff))
        with open('./DataHelp/motifs_samples_f1_forMSE.pickle', 'wb') as f:
            pickle.dump(X_sample_f1, f)
        with open('./DataHelp/motifs_samples_f3_forMSE.pickle', 'wb') as f:
            pickle.dump(X_sample_f3, f)
    else:
        X_sample = dict()
        for method in methods:
            s_max = 1 if method == RecursiveModularity else 10
            name_of_method = str(method).split('.')[-1].split("'")[0]
            X_sample[name_of_method] = dict()
            for number_of_nodes in list(range(l, r, step)):
                X_sample[name_of_method]['Number of nodes: ' + str(number_of_nodes)] = np.zeros(
                    (len(graphs) * s_max, len(names_of_all_motifs)))
                for s in range(s_max):
                    for i, (ds_name, gr) in enumerate(graphs):
                        my_dict_f1 = motifs_methods_f1[name_of_method]['Number of nodes: ' + str(number_of_nodes)][
                            ds_name + '_' + str(s)]
                        my_dict_f3 = motifs_methods_f3[name_of_method]['Number of nodes: ' + str(number_of_nodes)][
                            ds_name + '_' + str(s)]
                        X_sample[name_of_method]['Number of nodes: ' + str(number_of_nodes)][int(i+s*len(graphs))] = list(
                            map(lambda x: my_dict_f3[x] / my_dict_f1[x] if x in my_dict_f3 else 0, names_of_all_motifs))

        with open('./DataHelp/motifs_samples_forMSE.pickle', 'wb') as f:
            pickle.dump(X_sample, f)
    if diff_types:
        return X_full_f1,X_full_f3,X_sample_f1,X_sample_f3
    else:
        return X_full,X_sample


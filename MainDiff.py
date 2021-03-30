import pandas as pd
import numpy as np
import pickle

from multiprocessing import Pool
from modules.mse_funcs import find_MSE, plot

from modules.Modularity import RecursiveModularity

from datetime import datetime
import matplotlib.pyplot as plt

from modules.support_functions import Utils
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from modules.Motifs import Motifs

from littleballoffur import DegreeBasedSampler, \
    PageRankBasedSampler, \
    RandomEdgeSampler, \
    SnowBallSampler, \
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
    # Random-Walks Based
    RandomWalkSampler,
    RandomWalkWithJumpSampler,
    MetropolisHastingsRandomWalkSampler,
    NonBackTrackingRandomWalkSampler,
    CirculatedNeighborsRandomWalkSampler,
    LoopErasedRandomWalkSampler,
    RecursiveModularity
]
# global var
ms_max = 6
num_workers = 30
l = 10
r = 60
step = 10

# reading graphs from files
with open('all_graphs.pickle', 'rb') as f:
    graphs = pickle.load(f)

# Собственно, подсчет мотивов
X_full_f1, X_full_f3, X_sample_f1, X_sample_f3 = Motifs(True, graphs, ms_max, num_workers, l, r, step, methods)
X_full, X_sample = Motifs(False, graphs, ms_max, num_workers, l, r, step, methods)

print('MSE counting')

MSE_methods_f1 = dict()
MSE_methods_f3 = dict()
MSE_methods_nodif = dict()

for method in methods:
    d = datetime.now()
    name_of_method = str(method).split('.')[-1].split("'")[0]
    MSE_methods_f1.setdefault(name_of_method, dict())
    MSE_methods_f3.setdefault(name_of_method, dict())
    MSE_methods_nodif.setdefault(name_of_method, dict())
    # here is a parallelization

    inp = zip([method] * int((r - l) / step), list(range(l, r, step)))
    with Pool(num_workers) as executor:
        res = executor.map(lambda x: find_MSE(x, X_full, X_full_f1, X_full_f3, X_sample_f1[name_of_method],
                                              X_sample_f3[name_of_method], X_sample[name_of_method], graphs=graphs),
                           inp)

    for number_of_nodes, MSE_f1, MSE_f3, MSE_nodif in res:
        MSE_methods_f1[name_of_method][str(number_of_nodes)] = MSE_f1
        MSE_methods_f3[name_of_method][str(number_of_nodes)] = MSE_f3
        MSE_methods_nodif[name_of_method][str(number_of_nodes)] = MSE_nodif

    print(datetime.now() - d)

with open('./DataHelp/MSE_methods_f1.pickle', 'wb') as f:
    pickle.dump(MSE_methods_f1, f)
with open('./DataHelp/MSE_methods_f3.pickle', 'wb') as f:
    pickle.dump(MSE_methods_f3, f)
with open('./DataHelp/MSE_methods_nodif.pickle', 'wb') as f:
    pickle.dump(MSE_methods_nodif, f)

for name in MSE_methods_f1:
    name_of_graph = name.split("'")[0] + ' Motifs of different types. F1'
    plot(MSE_methods_f1[name], name_of_graph)
for name in MSE_methods_f3:
    name_of_graph = name.split("'")[0] + ' Motifs of different types. F3'
    plot(MSE_methods_f3[name], name_of_graph)
for name in MSE_methods_nodif:
    name_of_graph = name.split("'")[0] + ' Not different types of motifs'
    plot(MSE_methods_nodif[name], name_of_graph)
plt.figure(figsize=(10, 6))
mean_MSEs = []
for name in MSE_methods_f1:
    MSE_dict = MSE_methods_f1[name]
    MSE = pd.DataFrame(MSE_dict, columns=list(MSE_dict.keys()))
    y = list(MSE.mean())
    x = list(map(lambda x: int(x), list(MSE.columns)))
    ax = plt.scatter(x=x, y=y)
    plt.yscale('log')
    mean_MSEs.append(sum(y) / len(y))

plt.legend(['mean MSE. Motifs of different types. F1' + str(x[0]).split('.')[-1].split("'")[0] + ': ' + str(
    np.round(x[1], decimals=3)) for x in
            zip(methods, mean_MSEs)])
plt.show()
plt.figure(figsize=(10, 6))
mean_MSEs = []
for name in MSE_methods_f3:
    MSE_dict = MSE_methods_f3[name]
    MSE = pd.DataFrame(MSE_dict, columns=list(MSE_dict.keys()))
    y = list(MSE.mean())
    x = list(map(lambda x: int(x), list(MSE.columns)))
    ax = plt.scatter(x=x, y=y)
    plt.yscale('log')
    mean_MSEs.append(sum(y) / len(y))

plt.legend(['mean MSE. Motifs of different types. F3' + str(x[0]).split('.')[-1].split("'")[0] + ': ' + str(
    np.round(x[1], decimals=3)) for x in zip(methods, mean_MSEs)])
plt.show()

plt.figure(figsize=(10, 6))
mean_MSEs = []
for name in MSE_methods_nodif:
    MSE_dict = MSE_methods_nodif[name]
    MSE = pd.DataFrame(MSE_dict, columns=list(MSE_dict.keys()))
    y = list(MSE.mean())
    x = list(map(lambda x: int(x), list(MSE.columns)))
    ax = plt.scatter(x=x, y=y)
    plt.yscale('log')
    mean_MSEs.append(sum(y) / len(y))

plt.legend(['mean MSE. Not different types of motifs.' + str(x[0]).split('.')[-1].split("'")[0] + ': ' + str(
    np.round(x[1], decimals=3)) for x in zip(methods, mean_MSEs)])
plt.show()

print('Regression')

with Pool(num_workers) as executor:
    res = executor.map(Utils.count, list(zip(*graphs))[1])

y = []
for n_iter in res:
    y.append(n_iter)
with open('DataHelp/names_of_all_motifs.pickle', 'rb') as f:
    names_of_all_motifs = pickle.load(f)

with open('DataHelp/names_of_all_motifs_diff.pickle', 'rb') as f:
    names_of_all_motifs_diff = pickle.load(f)

# тк 10 раз повторяли граф для стохастических методов, то теперь надо обрезать, чтоб сохранлся один вариант распределения для одного графа
X_sample_to_save = dict(
    map(lambda e: (e[0], dict(map(lambda o: (o[0], o[1][:len(graphs)]), e[1].items()))), X_sample.items()))
X_sample_f3_to_save = dict(
    map(lambda e: (e[0], dict(map(lambda o: (o[0], o[1][:len(graphs)]), e[1].items()))), X_sample_f3.items()))
X_sample_f1_to_save = dict(
    map(lambda e: (e[0], dict(map(lambda o: (o[0], o[1][:len(graphs)]), e[1].items()))), X_sample_f1.items()))

for method in methods:
    name_of_method = str(method).split('.')[-1].split("'")[0]
    for n in list(range(l, r, step)):
        X_f1 = X_sample_f1_to_save[name_of_method]['Number of nodes: ' + str(n)]
        X_train, X_test, y_train, y_test = train_test_split(X_f1, y, test_size=0.3)
        X_train = pd.DataFrame(X_train, columns=names_of_all_motifs_diff)
        # Initialize CatBoostRegressor
        model = CatBoostRegressor(iterations=100, silent=True)
        # Fit model
        model.fit(X_train, y_train)
        # Get predictions
        preds = model.predict(X_test)
        # SHAP explainer:
        # explainer = shap.Explainer(model)
        # shap_values = explainer(X_train)
        # shap.plots.beeswarm(shap_values)
        print('Motifs of different types, F1. Method: ', name_of_method, ' Number of nodes: ' + str(n), ' MAPE ',
              Utils.mean_absolute_percentage_error(y_test, preds))

        X_f3 = X_sample_f3_to_save[name_of_method]['Number of nodes: ' + str(n)]
        X_train, X_test, y_train, y_test = train_test_split(X_f1, y, test_size=0.3)
        X_train = pd.DataFrame(X_train, columns=names_of_all_motifs_diff)
        # CatBoostRegressor
        model = CatBoostRegressor(iterations=100, silent=True)
        model.fit(X_train, y_train)
        # Get predictions
        preds = model.predict(X_test)
        # SHAP explainer:
        # explainer = shap.Explainer(model)
        # shap_values = explainer(X_train)
        # shap.plots.beeswarm(shap_values)
        print('Motifs of different types. F3. Method: ', name_of_method, ' Number of nodes: ' + str(n), ' MAPE ',
              Utils.mean_absolute_percentage_error(y_test, preds))

        X = X_sample_to_save[name_of_method]['Number of nodes: ' + str(n)]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        X_train = pd.DataFrame(X_train, columns=names_of_all_motifs)
        # CatBoostRegressor
        model = CatBoostRegressor(iterations=100, silent=True)
        model.fit(X_train, y_train)
        # Get predictions
        preds = model.predict(X_test)
        # explainer = shap.Explainer(model)
        # shap_values = explainer(X_train)
        # summarize the effects of all the features
        # shap.plots.beeswarm(shap_values)
        print('Motifs. Not different types. Method: ', name_of_method, ' Number of nodes: ' + str(n), ' MAPE ',
              Utils.mean_absolute_percentage_error(y_test, preds))

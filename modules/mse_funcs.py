from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_MSE(inp, X, X_f1, X_f3, X_samples_f1, X_samples_f3,
             X_samples, graphs):  #возвращает MSE для мотивов f1 И f3. Без разделения на разные типы мотивов. Размеры мотивов 3 и 4
    method, number_of_nodes = inp
    MSE_f1 = []
    MSE_f3 = []
    MSE_nodif = []

    for i, graph in enumerate(graphs):
        if number_of_nodes <= graph[1].number_of_nodes():
            motifs = X_f1[i]
            motifs_disjoint = X_f3[i]
            motifs_nodif = X[i]
            motifs_sample_con = X_samples_f1['Number of nodes: ' + str(number_of_nodes)][i]
            motifs_disjoint_sample_con = X_samples_f3['Number of nodes: ' + str(number_of_nodes)][i]
            motifs_nodif_sample = X_samples['Number of nodes: ' + str(number_of_nodes)][i]
            MSE_f1.append(mean_squared_error(motifs, motifs_sample_con))
            MSE_f3.append(mean_squared_error(motifs_disjoint, motifs_disjoint_sample_con))
            MSE_nodif.append(mean_squared_error(motifs_nodif, motifs_nodif_sample))
        else:
            MSE_f1.append(0)
            MSE_f3.append(0)
            MSE_nodif.append(0)
    return number_of_nodes, MSE_f1, MSE_f3, MSE_nodif


def plot(MSE_dict, name_of_method):
    MSE = pd.DataFrame(MSE_dict, columns=list(MSE_dict.keys()))
    plt.figure(figsize=(20, 6))

    plt.suptitle(name_of_method, fontsize=22)
    plt.subplot(121)
    plt.xlabel("number of nodes")
    plt.ylabel("MSE")
    g1 = sns.boxplot(data=MSE)
    g1.set_yscale('log')
    plt.subplot(122)
    plt.xlabel("number of nodes")
    plt.ylabel("MSE")
    y = list(MSE.mean())
    x = list(map(lambda x: int(x), list(MSE.columns)))
    g2 = sns.scatterplot(x=x, y=y)
    g2.set_yscale('log')
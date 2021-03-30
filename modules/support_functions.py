import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import numpy as np
import os
import pickle


class Utils:
    @staticmethod
    def motifs_to_vec(motifs1, motifs2):
        import numpy as np
        names_of_all_motifs = []  # list of all motif types in all datasets
        for name_of_motif in motifs1:
            if name_of_motif not in names_of_all_motifs:
                names_of_all_motifs.append(str(name_of_motif))
        for name_of_motif in motifs2:
            if name_of_motif not in names_of_all_motifs:
                names_of_all_motifs.append(str(name_of_motif))
        names_of_all_motifs = sorted(names_of_all_motifs)
        new_motifs = np.zeros((2, len(names_of_all_motifs)))
        sum_motif_1 = 0
        sum_motif_2 = 0
        for motif in motifs1:
            sum_motif_1 += motifs1[motif]
        for motif in motifs2:
            sum_motif_2 += motifs2[motif]
        for k, m in enumerate(names_of_all_motifs):
            if m in motifs1:
                new_motifs[0][k] = motifs1[m] / sum_motif_1
            if m in motifs2:
                new_motifs[1][k] = motifs2[m] / sum_motif_2
        return new_motifs[0], new_motifs[1]

    @staticmethod
    def find_motifs_all_types(ms_max, g):
        from SuperNoder.manager import Manager as Manager

        arg_tn = 'undirect'
        arg_th = '1'
        arg_m = 'h1'
        arg_ss = '100'
        arg_h1tr = 1

        distributions = {}
        distributions_disjoint = {}
        for arg_ms in range(3, ms_max + 1):
            argv = [' ', '-g', g[1], '-th', arg_th, '-ms', arg_ms, '-m', arg_m, \
                    '-h1tr', arg_h1tr, '-ss', arg_ss]
            path1 = './DataHelp/motifs_' + str(g[0]) + '_' + str(arg_ms) + 'size.pickle'
            path2 = './DataHelp/motifs_' + str(g[0]) + '_' + str(arg_ms) + 'size_disjoint.pickle'
            if os.path.exists(path1) and os.path.exists(path2):
                with open(path1, 'rb') as f:
                    distribution_f1 = pickle.load(f)
                with open(path2, 'rb') as f:
                    distribution_disjoint = pickle.load(f)
            else:
                m = Manager(argv)
                distribution_f1 = m.run()
                distribution_disjoint = m.disjoint_finder()
                with open(path1, 'wb') as f:
                    pickle.dump(distribution_f1, f)
                with open(path2, 'wb') as f:
                    pickle.dump(pickle.dump(distribution_f1, f), f)

            distributions = dict(list(distributions.items()) + list(distribution_f1.items()))
            distributions_disjoint = dict(
                list(distributions_disjoint.items()) + list(distribution_disjoint.items()))
        return g[
                   0], distributions, distributions_disjoint  # первое значение словаря - тип мотива или размер мотива, второе значение - количество таких мотивов в графе

    @staticmethod
    def find_motifs_diff_types(g, ms_max):
        from SuperNoder_diff_types.manager import Manager as Manager_types
        arg_tn = 'undirect'
        arg_th = '1'
        arg_m = 'h1'
        arg_ss = '100'
        arg_h1tr = 1

        distributions = {}
        distributions_disjoint = {}
        print(ms_max, g)
        for arg_ms in range(3, ms_max + 1):
            argv = [' ', '-g', g[1], '-th', arg_th, '-ms', arg_ms, '-m', arg_m, \
                    '-h1tr', arg_h1tr, '-ss', arg_ss]
            path1 = './DataHelp/motifs_' + str(g[0]) + '_' + str(arg_ms) + 'size_diff.pickle'
            path2 = './DataHelp/motifs_' + str(g[0]) + '_' + str(arg_ms) + 'size_diff_disjoint.pickle'
            if os.path.exists(path1) and os.path.exists(path2):
                with open(path1, 'rb') as f:
                    distribution_f1 = pickle.load(f)
                with open(path2, 'rb') as f:
                    distribution_disjoint = pickle.load(f)
            else:
                m = Manager_types(argv)
                distribution_f1 = m.run()
                distribution_disjoint = m.disjoint_finder()
                with open(path1, 'wb') as f:
                    pickle.dump(distribution_f1, f)
                with open(path2, 'wb') as f:
                    pickle.dump(distribution_disjoint, f)

            distributions = dict(list(distributions.items()) + list(distribution_f1.items()))
            distributions_disjoint = dict(list(distributions_disjoint.items()) + list(distribution_disjoint.items()))

        return g[
                   0], distributions, distributions_disjoint  # первое значение словаря - тип мотива или размер мотива, второе значение - количество таких мотивов в графе

    @staticmethod
    def count(g, beta=0.2, percentage_infected=0.01, estimations=10):
        len_nodes = len(g.nodes())
        list_of_iter = []

        for i in range(estimations):
            model = ep.SIModel(g)
            cfg = mc.Configuration()
            cfg.add_model_parameter('beta', beta)
            cfg.add_model_parameter("percentage_infected", percentage_infected)
            model.set_initial_status(cfg)

            iteration = model.iteration()  # initialization

            while (iteration['node_count'][1] < len_nodes):
                iteration = model.iteration()

            list_of_iter.append(iteration['iteration'])
        return np.mean(list_of_iter)

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

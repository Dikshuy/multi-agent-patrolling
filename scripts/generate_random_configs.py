#!/bin/usr/env python3

'''
A script to generate configs based on the parameters required

Output:
appropriate yaml files in config directory
'''

import networkx as nx
import sys
import rospkg
import random as rn
import tpbp_functions as fn

if __name__ == '__main__':
    dir_name = rospkg.RosPack().get_path('mrpp_sumo')
    if len(sys.argv[1:]) == 0:
        graph_name = ['complex_final','grid_5_5','first','grid_5_5_final']

        multiplicity = 1

        algo_name = 'without_intent_cr'
        vel = 10.
        # prior_nodes = rn.sample(graph.nodes(), num_priority)
        # prior_nodes = ['0', '4', '20', '24']
        # min_tp = fn.compute_min_tp(graph, prior_nodes)/vel

        # min_tp = 40.

        # lambda_priors = [0.0]
        # len_walks = [40, 80, 120]
        # max_divisions = 10
        # eps_prob = 0
        init_bots = [1, 3, 5, 7]
        no_of_deads = [0,7,12,18,25]
        no_runs = 3

        sim_length = 30000
        # discount_factors = [1]
        i = 0
        for _ in range(multiplicity):
            for g in graph_name:
                for ib in init_bots:
                    for deads in no_of_deads:
                        for run_id in range(no_runs):
                            graph = nx.read_graphml(dir_name + '/graph_ml/' + g + '.graphml')
                            i += 1
                            with open(dir_name + '/config/without_intent_cr/without_intent_cr{}.yaml'.format(i), 'w') as f:
                                f.write('use_sim_time: true\n')
                                f.write('graph: {}\n'.format(g))
                                # f.write('priority_nodes: {}\n'.format(' '.join(prior_nodes)))
                                # f.write('time_periods: {}\n'.format(' '.join(list(map(str, [min_tp] * num_priority)))))
                                # f.write('lambda_priority: {}\n'.format(l))
                                # f.write('length_walk: {}\n'.format(w))
                                # f.write('max_divisions: {}\n'.format(max_divisions))
                                # f.write('eps_prob: {}\n'.format(eps_prob))
                                f.write('init_bots: {}\n'.format(ib))
                                f.write('init_locations: \'\'\n')
                                f.write('done: false\n')
                                f.write('sim_length: {}\n'.format(sim_length))
                                # f.write('discount_factor: {}\n'.format(d))
                                f.write('random_string: without_intent_cr{}\n'.format(i))
                                f.write('algo_name: {}\n'.format(algo_name))
                                f.write('no_of_deads: {}\n'.format(deads))
                                f.write('run_id: {}'.format(run_id))
                            print (i)
    else:
        print ('Please pass the appropriate arguments')
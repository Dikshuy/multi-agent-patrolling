#!/usr/bin/env python3

import configparser as CP
import rospkg 
dir_name = rospkg.RosPack().get_path('tpbp_versions')
import networkx as nx
import numpy as np
import pandas as pd
import glob
configs = glob.glob(dir_name +  '/config_future_idle_*_0.txt')
graph_name = 'grid_5_5'
graph = nx.read_graphml(dir_name + '/graph_ml/' + graph_name + '.graphml')
nodes = graph.nodes()
edges = graph.edges()
cols_e = ['time']
for e in edges:
    cols_e.append(graph[e[0]][e[1]]['name'])
cols_n = ['time']
cols_n.extend(nodes)
for conf in configs:
    c = CP.ConfigParser()
    c.read(conf)
    sim_sets = c.sections()
    for s in sim_sets:
        print  (conf + ' ' + s)
        alg_name = c.get(section = s, option = 'algo').split('_')
        alg_name = '_'.join(alg_name[1:])
        params = list(map(float, c.get(section = s, option = 'algo_params').split(' ')))
        data_name = dir_name + '/' + s + '/sim_data_' + alg_name + '_{}_{}.in'.format(str(int(params[1])), str(int(params[2])))
        df_n = pd.DataFrame(columns = cols_n)
        df_e = pd.DataFrame(columns = cols_e)
        i = 0
        cur_time = 0.
        cur_data_n = {}
        cur_data_e = {}
        for n in df_n.columns:
            cur_data_n[n] = cur_time
        for e in df_e.columns:
            cur_data_e[e] = cur_time
        with open(data_name) as f:
            robots = {}
            for l in f.readlines():
                i += 1
                if i % 3 == 1:
                    next_time = float(l)
                    while cur_time < next_time:                
                        df_n = df_n.append(cur_data_n, ignore_index = True)
                        df_e = df_e.append(cur_data_e, ignore_index = True)
                        cur_time += 1.
                        cur_data_n['time'] = cur_time
                        cur_data_e['time'] = cur_time
                        for n in nodes:
                            cur_data_n[n] += 1.
                elif i % 3 == 2:
                    cur_nodes = l[:-1].split(' ')
                    for n in cur_nodes:
                        cur_data_n[n] = 0.
                else:
                    cur_robots = l[:-1].split(' ')
                    # print (cur_nodes, cur_robots)    
                    for r in range(len(cur_robots)):
                        if not cur_robots[r] in robots.keys():
                            robots[cur_robots[r]] = cur_nodes[r]
                        else:
                            try:
                                cur_data_e[graph[robots[cur_robots[r]]][cur_nodes[r]]['name']] += 1
                                robots[cur_robots[r]] = cur_nodes[r]
                            except:
                                pass
                                # print ('No such edge')
        df_n.to_csv(dir_name + '/' + s + '/idle_data_' + alg_name + '_{}_{}.csv'.format(str(int(params[1])), str(int(params[2]))))
        df_e.to_csv(dir_name + '/' + s + '/edge_data_' + alg_name + '_{}_{}.csv'.format(str(int(params[1])), str(int(params[2]))))
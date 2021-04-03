#!/bin/usr/env python3

import pandas as pd
import numpy as np
import rospkg
dir_name = rospkg.RosPack().get_path('tpbp_versions')
import glob
import networkx as nx
import configparser as CP

graph_name = 'grid_5_5'
graph = nx.read_graphml(dir_name + '/graph_ml/' + graph_name + '.graphml')
nodes = list(graph.nodes())
col_nodes = nodes[:].extend(['algo', 'algo_params', 'num_robots', 'priority_nodes', 'time_period'])
cols = ['algo', 'algo_params', 'num_robots', 'priority_nodes', 'time_period', 'avg_idle', 'max_idle', 'prior_avg', 'prior_max']
df = pd.DataFrame(columns = cols)
confs = glob.glob(dir_name + '/config_*.txt')
df_nodes = pd.DataFrame(columns = col_nodes)
for conf in confs:
    c = CP.ConfigParser()
    c.read(conf)
    sim_sets = c.sections()
    for s in sim_sets:
        data_nodes = {}
        data_nodes['algo'] = c.get(section = s, option = 'algo')
        data_nodes['algo_params'] = c.get(section = s, option = 'algo_params')
        data_nodes['num_robots'] = c.get(section = s, option = 'num_robots')
        data_nodes['priority_nodes'] = c.get(section = s, option = 'priority_nodes')
        data_nodes['time_period'] = c.get(section = s, option = 'time_period')
        
        data = {}
        data['algo'] = c.get(section = s, option = 'algo')
        data['algo_params'] = c.get(section = s, option = 'algo_params')
        data['num_robots'] = c.get(section = s, option = 'num_robots')
        data['priority_nodes'] = c.get(section = s, option = 'priority_nodes')
        data['time_period'] = c.get(section = s, option = 'time_period')

        alg_name = data['algo'].split('_')
        alg_name = '_'.join(alg_name[1:])
        params = list(map(float, data['algo_params'].split(' ')))
        file_name = dir_name + '/' + s + '/' + 'idle_data_{}_{}_{}.csv'.format(alg_name, str(int(params[1])), str(int(params[2])))
        df_temp = pd.read_csv(file_name, index_col = 0)
        prior_nodes = data['priority_nodes'].split(' ')
        df_temp['prior_avg'] = df_temp[prior_nodes].mean(axis = 1)
        df_temp['prior_max'] = df_temp[prior_nodes].max(axis = 1)

        df_temp['avg_idle'] = df_temp[list(nodes)].mean(axis = 1)
        df_temp['max_idle'] = df_temp[list(nodes)].max(axis = 1)
        df_temp.to_csv(file_name)
        data_nodes1 = data_nodes.copy()
        data_nodes2 = data_nodes.copy()
        data_nodes1['idle'] = 'Mean'
        data_nodes1.update(dict(df_temp[list(graph.nodes())].mean(axis = 0)))
        data_nodes2['idle'] = 'Max'
        data_nodes2.update(dict(df_temp[list(graph.nodes())].max(axis = 0)))
        data['avg_idle'] = df_temp['avg_idle'].iloc[3600:].mean()
        data['max_idle'] = df_temp['max_idle'].iloc[3600:].max()
        data['prior_avg'] = df_temp['prior_avg'].iloc[3600:].mean()
        data['prior_max'] = df_temp['prior_max'].iloc[3600:].max()
        
        df_nodes = df_nodes.append(data_nodes1, ignore_index = True)
        df_nodes = df_nodes.append(data_nodes2, ignore_index = True)
        df = df.append(data, ignore_index = True)
        print (conf + ' ' + s)

df_nodes.to_csv(dir_name + '/all_idles.csv')
df.to_csv(dir_name + '/summary_all_nodes.csv')
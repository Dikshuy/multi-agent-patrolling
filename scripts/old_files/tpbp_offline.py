#!/usr/bin/env python

import os
import networkx as nx
from copy import deepcopy
#from itertools import imap

def add_vertex_trail(graph, path, len_path, vertex, dest, len_max):
    # Distinct edges

    cur = path[-1]
    len_rem = nx.shortest_path_length(graph, cur, dest, weight = 'length')
    #print "here----", len_path, len_rem, len_max
    if (len_rem + len_path) > len_max: #Unsure about  this
        return False
    if not cur in path[:-1]:
        return True
    for i in range(len(path[:-2])):
        if path[i] == cur and path[i + 1] == vertex:
            return False
    return True

def add_vertex_path(graph, path, len_path, vertex, dest, len_max):
    # Distinct vertices
    cur = path[-1]
    if vertex in path and vertex != dest:
        return False
    len_rem = nx.shortest_path_length(graph, cur, dest)
    if (len_rem + len_path) > len_max:
        return False
    return True

def add_vertex_walk(graph, path, len_path, vertex, dest, len_max):
    # All walks
    cur = path[-1]
    len_rem = nx.shortest_path_length(graph, cur, dest)
    if (len_rem + len_path) > len_max:
        return False
    return True

def compute_valid_trails(graph, source, dest, len_max, folder):
    # Distinct edges
    with open(folder + '/valid_trails_{}_{}_{}.in'.format(str(source), str(dest),str(int(len_max))), 'a+') as f:
        with open(folder + '/vp_temp_{}.in'.format(0), 'w') as f1:
            f1.write(str(source) + ' ' + str(0) + '\n')

        count = 1
        steps = 0
        while count != 0:
            count = 0
            with open(folder + '/vp_temp_{}.in'.format(steps), 'r') as f0:
                with open(folder + '/vp_temp_{}.in'.format(steps + 1), 'w') as f1:
                    for line in f0:
                        line1 = line.split('\n')
                        line_temp = line1[0]
                        line1 = line_temp.split(' ')
                        # print("jhere", line1)
                        path = list(map(str, line1[:-1]))
                        # print('here', path)
                        len_path = float(line1[-1])
                        neigh = graph.neighbors(path[-1])

                        for v in neigh:
                            if add_vertex_trail(graph, path, len_path, v, dest, len_max * 10.):
                                temp = ' '.join(line1[:-1])
                                temp = temp + ' ' + str(v)
                                if v == dest:
                                    f.write(temp + '\n')
                                else:
                                    count += 1
                                    temp += ' ' + str(graph[path[-1]][v]['length'] + len_path)
                                    f1.write(temp + '\n')
            steps += 1

    for i in range(steps + 1):
        os.remove(folder + '/vp_temp_{}.in'.format(i))

def compute_valid_paths(graph, source, dest, len_max, folder):
    # Distinct vertices
    with open(folder + '/valid_paths_{}_{}_{}.in'.format(str(source), str(dest), str(int(len_max))), 'a+') as f:
        with open(folder + '/vp_temp_{}.in'.format(0), 'w') as f1:
            f1.write(str(source) + ' ' + str(0) + '\n')

        count = 1
        steps = 0
        while count != 0:
            count = 0
            with open(folder + '/vp_temp_{}.in'.format(steps), 'r') as f0:
                with open(folder + '/vp_temp_{}.in'.format(steps + 1), 'w') as f1:
                    for line in f0:
                        line1 = line.split('\n')
                        line_temp = line1[0]
                        line1 = line_temp.split(' ')
                        path = list(map(str, line1[:-1]))
                        len_path = float(line1[-1])
                        neigh = graph.neighbors(path[-1])
                        for v in neigh:
                            if add_vertex_path(graph, path, len_path, v, dest, len_max):
                                temp = ' '.join(line1[:-1])
                                temp = temp + ' ' + str(v)
                                if v == dest:
                                    f.write(temp + '\n')
                                else:
                                    count += 1
                                    temp += ' ' + str(graph[path[-1]][v]['length'] + len_path)
                                    f1.write(temp + '\n')
            steps += 1

    for i in range(steps + 1):
        os.remove(folder + '/vp_temp_{}.in'.format(i))

def compute_valid_walks(graph, source, dest, len_max, folder):
    #All walks
    # print 'olo'
    with open(folder + '/valid_walks_{}_{}_{}.in'.format(str(source), str(dest),str(int(len_max))), 'a+') as f:
        # print 'olo2'
        with open(folder + '/vp_temp_{}.in'.format(0), 'w') as f1:
            f1.write(str(source) + ' ' + str(0) + '\n')
        count = 1
        steps = 0
        while count != 0:
            count = 0
            with open(folder + '/vp_temp_{}.in'.format(steps), 'r') as f0:
                with open(folder + '/vp_temp_{}.in'.format(steps + 1), 'w') as f1:
                    for line in f0:
                        line1 = line.split('\n')
                        line_temp = line1[0]
                        line1 = line_temp.split(' ')
                        path = list(map(str, line1[:-1]))
                        len_path = float(line1[-1])
                        neigh = graph.neighbors(path[-1])
                        for v in neigh:
                            if add_vertex_walk(graph, path, len_path, v, dest, len_max):
                                temp = ' '.join(line1[:-1])
                                temp = temp + ' ' + str(v)
                                if v == dest:
                                    f.write(temp + '\n')
                                else:
                                    count += 1
                                    temp += ' ' + str(graph[path[-1]][v]['length'] + len_path)
                                    f1.write(temp + '\n')
            steps += 1

    for i in range(steps + 1):
        os.remove(folder + '/vp_temp_{}.in'.format(i))

def all_valid_trails(graph, node_set, len_max, folder):
    for i in range(len(node_set)):
        for j in range(len(node_set)):
            compute_valid_trails(graph, node_set[i], node_set[j], len_max[i], folder)

def all_valid_paths(graph, node_set, len_max, folder):
    for i in range(len(node_set)):
        for j in range(len(node_set)):
            compute_valid_paths(graph, node_set[i], node_set[j], len_max[i], folder)

def all_valid_walks(graph, node_set, len_max, folder):
    for i in range(len(node_set)):
        for j in range(len(node_set)):
            compute_valid_walks(graph, node_set[i], node_set[j], len_max[i], folder)

def is_node_interior(graph, source, dest, node):
    path_s = nx.dijkstra_path(graph, source, node, 'length')
    g_temp = deepcopy(graph)
    g_temp.remove_nodes_from(path_s[1:-1])
    if nx.is_weakly_connected(g_temp):
        return True
    if source != dest:
        path_d = nx.dijkstra_path(graph, node, dest, 'length')
        g_temp = deepcopy(graph)
        g_temp.remove_nodes_from(path_d[1:-1])
        if nx.is_weakly_connected(g_temp):
            return True
    return False

# def repeatable_vertices(graph, source, dest, node):
#     path_s = nx.shortest_path(graph, source, node)
#     path_d = nx.shortest_path(graph, dest, node)
#     repeat = []
#     for i in path_s:
#         if i in path_d:
#             repeat.append(i)
#             break
#     path_temp = path_s[path_s.index(repeat[0]):]
#     g_temp = deepcopy(graph)
#     g_temp.remove_nodes_from([path_temp[1:-1]])

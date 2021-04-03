#!/bin/usr/env python3

import networkx as nx
import numpy as np

def shortest_path_one_to_all(graph, source, label = 'weight'):
    node_values = {}
    node_paths = {}
    for i in graph.nodes():
        node_values[i] = np.inf
        node_paths[i] = []
    node_values[source] = 0.

    delta = 1e-5
    epsilon = np.inf

    while epsilon > delta:
        epsilon = 0.
        for i in graph.nodes():
            for j in graph.predecessors(i):
                if node_values[i] > node_values[j] + graph.get_edge_data(j, i)[label]:
                    epsilon += 1
                    node_values[i] = node_values[j] + graph.get_edge_data(j, i)[label]
                    node_paths[i] = node_paths[j][:]
                    node_paths[i].append(graph.get_edge_data(j, i)['name'])
    return (node_values, node_paths)

def shortest_path_all_to_one(graph, dest, label = 'weight'):
    node_values = {}
    node_paths = {}
    for i in graph.nodes():
        node_values[i] = np.inf
        node_paths[i] = []
    node_values[dest] = 0.

    delta = 1e-5
    epsilon = np.inf

    while epsilon > delta:
        epsilon = 0.
        for i in graph.nodes():
            for j in graph.predecessors(i):
                if node_values[j] > node_values[i] + graph.get_edge_data(j, i)[label]:
                    epsilon += 1
                    node_values[j] = node_values[i] + graph.get_edge_data(j, i)[label]
                    node_paths[j] = node_paths[i][:]
                    node_paths[j].insert(0, graph.get_edge_data(j, i)['name'])
    return (node_values, node_paths)


def compute_min_tp(graph, priority_nodes):
    #For complete patrol
    shortest_paths_from = {}
    shortest_paths_to = {}
    for i in priority_nodes:
        shortest_paths_from[i] = shortest_path_one_to_all(graph, i, 'length')
        shortest_paths_to[i] = shortest_path_all_to_one(graph, i, 'length')
    
    #To non-priority nodes
    longest_1 = 0.
    for i in graph.nodes():
        temp = np.inf
        for j in priority_nodes:
            if shortest_paths_from[j][0][i] + shortest_paths_to[j][0][i] < temp:
                temp = shortest_paths_from[j][0][i] + shortest_paths_to[j][0][i]
        if temp > longest_1:
            longest_1 = temp

    #To priority nodes
    longest_2 = 0.
    for i in priority_nodes:
        temp_priority = priority_nodes[:]
        temp_done = [i]
        temp_priority.remove(i)
        while len(temp_priority) > 0:
            temp_i = np.inf
            temp_n = i
            for j in temp_done:
                for k in temp_priority:
                    temp = shortest_paths_from[j][0][k]
                    print (j, k, temp)
                    if temp < temp_i:
                        temp_i = temp
                        temp_n = k
            print (temp_i)
            temp_done.append(temp_n)
            temp_priority.remove(temp_n)
            if longest_2 < temp_i:
                longest_2 = temp_i

    return max(longest_1, longest_2) 
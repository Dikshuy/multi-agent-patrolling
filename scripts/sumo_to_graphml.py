#!/bin/usr/env python3

import sys
import rospkg
import os
import xml.etree.ElementTree as et
import networkx as nx
import math

def graph_wrapper(graph_name ='grid_5_5'):
    
    dir_name = rospkg.RosPack().get_path('mrpp_sumo')
    sumo_folder = dir_name + '/graph_sumo'
    graphml_folder = dir_name + '/graph_ml'
    node_file = sumo_folder + '/{}.nod.xml'.format(graph_name)
    edge_file = sumo_folder + '/{}.edg.xml'.format(graph_name)
    graphml_file = graphml_folder + '/{}.graphml'.format(graph_name)

    sumo_to_graphml(node_file, edge_file, graphml_file)

    graph = nx.read_graphml(graphml_file)
    print ('Converted Successfully!')
    print ('Graph Name: {}'.format(graph_name))
    print ('Total Nodes: {}'.format(str(len(graph.nodes()))))
    print ('Total Edges: {}'.format(str(len(graph.edges()))))

def sumo_to_graphml(node_file, edge_file, graphml_file):

    graph = nx.DiGraph()

    node_parse = et.parse(node_file)
    node_root = node_parse.getroot()

    for n in node_root.iter('node'):
        graph.add_node(n.attrib['id'])
        graph.nodes[n.attrib['id']]['x'] = float(n.attrib['x'])
        graph.nodes[n.attrib['id']]['y'] = float(n.attrib['y'])

    edge_parse = et.parse(edge_file)
    edge_root = edge_parse.getroot()

    for e in edge_root.iter('edge'):
        n1 = e.attrib['from']
        n2 = e.attrib['to']
        graph.add_edge(n1, n2)
        graph[n1][n2]['name'] = e.attrib['id']
        temp1 = []
        temp2 = []
        temp1.append(float(graph.nodes[n1]['x']))
        temp2.append(float(graph.nodes[n1]['y']))
        temp = []
        if 'shape' in e.attrib.keys():
            temp = e.attrib['shape'].strip()
            temp = temp.split(' ')
            for i in range(len(temp)):
                co_ord = temp[i].split(',')
                temp1.append(float(co_ord[0]))
                temp2.append(float(co_ord[1]))
        temp1.append(float(graph.nodes[n2]['x']))
        temp2.append(float(graph.nodes[n2]['y']))
        length = 0
        for i in range(1, 2 + len(temp)):
            length += math.sqrt((temp1[i] - temp1[i - 1]) ** 2 + (temp2[i] - temp2[i - 1]) ** 2)
        graph[n1][n2]['length'] = length

    nx.write_graphml(graph, graphml_file)

def main(name):
    graph_wrapper(name)

if __name__ == '__main__':
    main(sys.argv[1])

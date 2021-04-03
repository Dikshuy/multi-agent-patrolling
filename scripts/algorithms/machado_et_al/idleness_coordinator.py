#!/usr/bin/env python3

'''
Idleness Coordinator
'''

import rospkg
import numpy as np
import rospy
import networkx as nx
from mrpp_sumo.srv import NextTaskBot, NextTaskBotResponse, AlgoReady, AlgoReadyResponse
from mrpp_sumo.msg import AtNode
import random as rn

class IC:

    def __init__(self, g):
        self.ready = False
        self.graph = g
        self.stamp = 0.
        self.robots = {}
        self.nodes = list(self.graph.nodes())
        for node in self.graph.nodes():
            self.graph.nodes[node]['idleness'] = 0.

        rospy.Service('algo_ready', AlgoReady, self.callback_ready)
        self.ready = True

    def callback_idle(self, data):
        if self.stamp < data.stamp:
            dev = data.stamp - self.stamp
            self.stamp = data.stamp
            for i in self.graph.nodes():
                self.graph.nodes[i]['idleness'] += dev
                
            for i, n in enumerate(data.node_id):
                self.graph.nodes[n]['idleness'] = 0.
                
    
    def callback_next_task(self, req):
        node = req.node_done
        t = req.stamp
        bot = req.name

        self.graph.nodes[node]['idleness'] = 0.

        
        idles = []
        for n in self.nodes:
            idles.append(self.graph.nodes[n]['idleness'])

        temp = self.nodes.copy()
        max_id = 0
        if len(self.nodes) > 1:
            max_ids = list(np.where(idles == np.amax(idles))[0])
            max_id = rn.sample(max_ids, 1)[0]
        dest_node = temp[max_id]
        while dest_node == node or dest_node in self.robots.values():
            if len(temp) == 1 and dest_node != node:
                dest_node = temp[0]
                break
            elif len(temp) == 1 and dest_node == node:
                temp1 = self.nodes.copy()
                temp1.remove(node)
                dest_node = rn.sample(temp1, 1)[0]
                break
            temp.remove(dest_node)
            idles.pop(max_id)
            max_ids = list(np.where(idles == np.amax(idles))[0])
            max_id = rn.sample(max_ids, 1)[0]
            dest_node = temp[max_id]
        next_walk = nx.dijkstra_path(g, node, dest_node, 'length')
        self.robots[bot] = dest_node
        next_departs = [t] * (len(next_walk) - 1)
        return NextTaskBotResponse(next_departs, next_walk)

    def callback_ready(self, req):
        algo_name = req.algo
        if algo_name == 'idleness_coordinator' and self.ready:
            return AlgoReadyResponse(True)
        else:
            return AlgoReadyResponse(False)
if __name__ == '__main__':
    rospy.init_node('ic', anonymous= True)
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    done = False
    graph_name = rospy.get_param('/graph')
    g = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')

    s = IC(g)

    rospy.Subscriber('at_node', AtNode, s.callback_idle)
    rospy.Service('bot_next_task', NextTaskBot, s.callback_next_task)

    done = False
    while not done:
        done = rospy.get_param('/done')
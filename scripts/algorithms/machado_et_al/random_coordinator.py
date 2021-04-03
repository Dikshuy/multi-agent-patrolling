#!/usr/bin/env python3

'''
Random Coordinator
'''

import rospkg
import numpy as np
import rospy
import networkx as nx
from mrpp_sumo.srv import NextTaskBot, NextTaskBotResponse
from mrpp_sumo.msg import AtNode
import random as rn

class RC:

    def __init__(self, g):
        self.ready = False
        self.graph = g
        self.stamp = 0.
        self.robots = {}
        self.nodes = list(self.graph.nodes())
        rospy.Service('algo_ready', AlgoReady, self.callback_ready)
        self.ready = True

    def callback_idle(self, data):
        pass
                
    
    def callback_next_task(self, req):
        node = req.node_done
        t = req.stamp
        bot = req.name

        temp = self.nodes.copy()
        dest_node = rn.sample(temp, 1)[0]
        while dest_node == node or dest_node in self.robots.values():
            if len(temp) == 1 and dest_node != node:
                dest_node = temp[0]
                break
            elif len(temp) == 1 and dest_node == node:
                temp = self.nodes.copy()
                temp.remove(node)
                dest_node = rn.sample(temp, 1)[0]
            temp.remove(dest_node)
            dest_node = rn.sample(temp, 1)[0]
        next_walk = nx.dijkstra_path(g, node, dest_node, 'length')
        self.robots[bot] = dest_node
        next_departs = [t] * (len(next_walk) - 1)
        return NextTaskBotResponse(next_departs, next_walk)
    
    def callback_ready(self, req):
        algo_name = req.algo
        if algo_name == 'random_coordinator' and self.ready:
            return AlgoReadyResponse(True)
        else:
            return AlgoReadyResponse(False)

if __name__ == '__main__':
    rospy.init_node('rc', anonymous= True)
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    done = False
    graph_name = rospy.get_param('/graph')
    g = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')

    s = RC(g)

    rospy.Subscriber('at_node', AtNode, s.callback_idle)
    rospy.Service('bot_next_task', NextTaskBot, s.callback_next_task)

    done = False
    while not done:
        done = rospy.get_param('/done')
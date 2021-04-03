#!/usr/bin/env python3


'''
RANDOM

ROS Params
'''
import rospy
import rospkg
import networkx as nx
from mrpp_sumo.srv import NextTaskBot, NextTaskBotResponse
from mrpp_sumo.srv import AlgoReady, AlgoReadyResponse
from mrpp_sumo.msg import AtNode 

import random as rn
class Random_MRPP:
    def __init__(self, graph):
        self.ready = False
        self.graph = graph
        self.stamp = 0.
        self.ready = True

    def callback_idle(self, data):
        #Update idleness of the nodes in the graph
        pass

    def callback_next_task(self, req):
        node = req.node_done
        t = req.stamp
        bot = req.name

        neigh = list(self.graph.successors(node))
        next_node = rn.sample(neigh, 1)[0]
        next_walk = [node, next_node]
        next_departs = [t]
        return NextTaskBotResponse(next_departs, next_walk)

    def callback_ready(self, req):
        algo_name = req.algo
        if algo_name == 'random_mrpp' and self.ready:
            return AlgoReadyResponse(True)
        else:
            return AlgoReadyResponse(False)

if __name__ == '__main__':
    rospy.init_node('random', anonymous = True)
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    done = False
    graph_name = rospy.get_param('/graph')
    g = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')
    
    s = Random_MRPP(g)
    rospy.Subscriber('at_node', AtNode, s.callback_idle)
    rospy.Service('bot_next_task', NextTaskBot, s.callback_next_task)
    rospy.Service('algo_ready', AlgoReady, s.callback_ready)
    while not done:
        done = rospy.get_param('/done')